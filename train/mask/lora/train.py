import os
import argparse
from pathlib import Path
import itertools
import time
from packaging import version
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from models.ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnMapProcessor
from diffusers.models import ControlNetModel
from diffusers.utils.import_utils import is_xformers_available
from models.ip_adapter.ip_adapter_faceid import ProjPlusModel

from train.loss import compute_total_loss
from dataset.MaskDataset import Dataset4ShortCaptionRandomText,collate_fn

class IPAdapter(torch.nn.Module):
    """IP-Adapter-FaceIDPlus"""
    def __init__(self, unet, image_proj_model, adapter_modules, contorlnet, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.contorlnet = contorlnet

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, face_id_embeds, image_embeds, controlnet_image):
        ip_tokens = self.image_proj_model(face_id_embeds, image_embeds, shortcut=True)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        
        down_block_res_samples, mid_block_res_sample = self.contorlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=ip_tokens,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )

        # Predict the noise residual
        
        noise_pred = self.unet(
            noisy_latents, 
            timesteps,
            encoder_hidden_states = encoder_hidden_states,
            down_block_additional_residuals=[
                        sample.to(dtype=encoder_hidden_states.dtype) for sample in down_block_res_samples
                    ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=encoder_hidden_states.dtype),
        ).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--face_id_dir",
        type=str,
        default="/home/ddwgroup/workplace/hq1M_face_square_faceidEmbed",
        required=True,
        help="face_id_embed",
    )
    parser.add_argument(
        "--control_img_dir",
        type=str,
        default="/home/ddwgroup/public-Datasets/HQFaceSquare400W/HQFaceSquare400W_KPS",
        required=True,
        help="control img dir",
    )
    parser.add_argument(
        "--parsing_mask_dir",
        type=str,
        default="/home/ubuntu/san/zyx/face_makeup_2/public-Datasets/HQFaceSquare400W/HQFaceSquare400W_Mask",
        required=True,
        help="parsing mask dir",
    )
    parser.add_argument(
        "--data_sample_size",
        type=int,
        default=None,
        help="Number of samples to randomly select from the dataset. If not specified, use entire dataset.",
    )
    parser.add_argument(
        "--data_sample_seed",
        type=int,
        default=42,
        help="Random seed for data sampling reproducibility.",
    )
    parser.add_argument(
        "--face_img_dir",
        type=str,
        default="/home/ddwgroup/public-Datasets/HQFaceSquare400W/HQFaceSquare400W_FaceAlign",
        required=True,
        help="face img dir",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xformers",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--mask_downsample_rate",
        type=int,
        default=16,
        help=(
            "mask downsample rate"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument(
        "--maploss_factor",
        type=float,
        default=1e-3,
        help="maploss factor",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10000, 
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    # 从预训练的控制网络中加载模型（第二次训练）
    # ControlNetModel.from_pretrained(args.controlnet_path)
    # 从Unet中加载控制网络（默认，初始
    controlnet = ControlNetModel.from_unet(unet)


    # freeze parameters of models to save more memory
    
    # for map loss, train unet
    # unet.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.train()
    # unet.train()
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print("xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details")
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    #ip-adapter
    image_proj_model = ProjPlusModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        id_embeddings_dim=512,
        clip_embeddings_dim=image_encoder.config.hidden_size,
        num_tokens=4
    )
    # init adapter modules
    lora_rank = 128
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = LoRAIPAttnMapProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            attn_procs[name].load_state_dict(weights, strict=False)
    unet.set_attn_processor(attn_procs)
    
    cross_attn_layers = [module for module in unet.modules() if isinstance(module, LoRAIPAttnMapProcessor)]

    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, controlnet, args.pretrained_ip_adapter_path)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters(), ip_adapter.contorlnet.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = Dataset4ShortCaptionRandomText(args.data_json_file,
                              tokenizer=tokenizer,
                              size=args.resolution,
                              image_root_path=args.data_root_path,
                              face_id_dir=args.face_id_dir,
                              face_img_dir=args.face_img_dir,
                              control_img_dir=args.control_img_dir,
                              parsing_mask_dir=args.parsing_mask_dir,
                              sample_size=args.data_sample_size,
                              sample_seed=args.data_sample_seed
                              )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,  
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    
    # Initialize training state
    global_step = 0
    first_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint is not None:
        if os.path.exists(args.resume_from_checkpoint):
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            
            # Extract global step from checkpoint directory name
            try:
                global_step = int(args.resume_from_checkpoint.split("-")[-1])
            except (ValueError, IndexError):
                print("Warning: Could not extract global_step from checkpoint name, starting from 0")
                global_step = 0
            
            # Calculate starting epoch based on global_step and dataset size
            steps_per_epoch = len(train_dataloader)
            first_epoch = global_step // steps_per_epoch
            steps_to_skip_in_epoch = global_step % steps_per_epoch
            
            # Validate the resume logic
            if global_step > 0:
                print(f"Resuming from checkpoint: step {global_step}, epoch {first_epoch}")
                print(f"Steps per epoch: {steps_per_epoch}")
                print(f"Will skip {steps_to_skip_in_epoch} steps in epoch {first_epoch}")
                print(f"Next checkpoint will be saved at step {((global_step // args.save_steps) + 1) * args.save_steps}")
            else:
                print("Starting training from beginning")
        else:
            print(f"Checkpoint directory {args.resume_from_checkpoint} not found. Starting from scratch.")
            global_step = 0
            first_epoch = 0
    
    for epoch in range(first_epoch, args.num_train_epochs):
        begin = time.perf_counter()
        
        # Handle skipping for resumed training
        steps_to_skip_in_epoch = 0
        if epoch == first_epoch and global_step > 0:
            steps_to_skip_in_epoch = global_step % len(train_dataloader)
            if steps_to_skip_in_epoch > 0:
                print(f"Using accelerator to skip {steps_to_skip_in_epoch} steps in epoch {first_epoch}...")
                # Use accelerator's built-in skip functionality
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, steps_to_skip_in_epoch)
                print(f"Skipping completed, starting training from step {steps_to_skip_in_epoch}")
            else:
                skipped_dataloader = train_dataloader
        else:
            skipped_dataloader = train_dataloader
        
        # Training loop with potentially skipped dataloader
        for step, batch in enumerate(skipped_dataloader):
            # Calculate the actual step number within the epoch
            actual_step_in_epoch = step + steps_to_skip_in_epoch
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(accelerator.device, dtype=weight_dtype)
                
                faceid_embeds = batch["face_id_embed"].to(accelerator.device, dtype=weight_dtype)

                clip_image_embeds = image_encoder(batch["clip_image_embed"].to(accelerator.device, dtype=weight_dtype) , output_hidden_states=True).hidden_states[-2]


                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]


                controlnet_image = batch["contorlnet_image"].to(accelerator.device, dtype=weight_dtype)

                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, faceid_embeds, clip_image_embeds, controlnet_image)

                # 得到注意力图
                attention_maps = [layer.attention_probs for layer in cross_attn_layers if hasattr(layer, 'attention_probs')]
                attention_maps = [map for map in attention_maps if map.shape[1]==(args.resolution // args.mask_downsample_rate)**2]
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                map_loss = compute_total_loss(attention_maps,batch)
                total_loss = loss + args.maploss_factor*map_loss
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss).mean().item()
                avg_map_loss = accelerator.gather(map_loss).mean().item()
                avg_total_loss = avg_loss + args.maploss_factor * avg_map_loss                
                
                # Backpropagate
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    # print(f"捕捉到 {len(attention_maps)} 个自注意力图")

                    print("Epoch {}, step {}, global_step {}, data_time: {:.4f}s, time: {:.4f}s, loss: {:.6f}, map_loss: {:.6f}, total_loss: {:.6f}".format(
                        epoch, actual_step_in_epoch, global_step, load_data_time, time.perf_counter() - begin, avg_loss, avg_map_loss, avg_total_loss))
                    
            # Update global step (we only reach here if we're processing the batch)
            global_step += 1
            
            # Debug: Print save step information periodically
            if accelerator.is_main_process and global_step % 1000 == 0:
                steps_until_save = args.save_steps - (global_step % args.save_steps)
                print(f"DEBUG: global_step={global_step}, save_steps={args.save_steps}, steps_until_save={steps_until_save}")
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                if accelerator.is_main_process:
                    print(f"SAVING checkpoint at step {global_step}: {save_path}")
                accelerator.save_state(save_path)
                if accelerator.is_main_process:
                    print(f"Successfully saved checkpoint at step {global_step}: {save_path}")
            
            begin = time.perf_counter()
            
if __name__ == "__main__":
    main()    
