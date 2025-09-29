import os
import torch
import PIL
from typing import List, Union, Optional, Dict, Any, Callable
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import logging
from safetensors.torch import safe_open
from .resampler import PerceiverAttention, FeedForward
from .attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
from .TokenModifiedPipeline import TokenModifiedPipeline
logger = logging.get_logger(__name__)


class FacePerceiverResampler(torch.nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)

class ProjPlusModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )
        
    def forward(self, id_embeds, clip_embeds, shortcut=False, scale=1.0):
        
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out

class FaceMakeUpPipeline(TokenModifiedPipeline):
    """Pipeline for generating face makeup using IP-Adapter, ControlNet and face embeddings."""
    
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        controlnet,
        scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        image_encoder=None,
        image_proj_model=None,
        lora_rank=128,
        num_tokens=4,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )
        
        self.register_modules(
            image_encoder=image_encoder,
            image_proj_model=image_proj_model
        )
        
        self.lora_rank = lora_rank
        self.num_tokens = num_tokens
        self.clip_image_processor = CLIPImageProcessor() if image_encoder is not None else None
        
        # Configure IP-Adapter if unet is initialized
        if self.unet is not None and hasattr(self, "image_proj_model") and self.image_proj_model is not None:
            self.set_ip_adapter()

    @classmethod
    def from_pretrained(cls, pretrained_model_path, image_encoder_path=None, ip_ckpt=None, 
                        controlnet_ckpt=None, lora_rank=128, num_tokens=4, **kwargs):
        """Load pipeline from pretrained models with additional IP-Adapter components."""
        pipeline = super().from_pretrained(pretrained_model_path, **kwargs)
        
        # Load image encoder if provided
        if image_encoder_path:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
            pipeline.register_modules(image_encoder=image_encoder)
            
            # Initialize image projection model
            image_proj_model = ProjPlusModel(
                cross_attention_dim=pipeline.unet.config.cross_attention_dim,
                id_embeddings_dim=512,
                clip_embeddings_dim=image_encoder.config.hidden_size,
                num_tokens=num_tokens,
            )
            pipeline.register_modules(image_proj_model=image_proj_model)
            
            # Set up IP-Adapter attention processors
            pipeline.lora_rank = lora_rank
            pipeline.num_tokens = num_tokens
            pipeline.clip_image_processor = CLIPImageProcessor()
            pipeline.set_ip_adapter()
        
        return pipeline
    
    def set_ip_adapter(self):
        """Configure IP-Adapter attention processors for UNet."""
        
        unet = self.unet
        attn_procs = {}
        
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
                attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, 
                    cross_attention_dim=cross_attention_dim, 
                    rank=self.lora_rank,
                ).to(self.device, dtype=self.dtype)
            else:
                attn_procs[name] = LoRAIPAttnProcessor(
                    hidden_size=hidden_size, 
                    cross_attention_dim=cross_attention_dim, 
                    scale=1.0, 
                    rank=self.lora_rank, 
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.dtype)
                
        unet.set_attn_processor(attn_procs)
    
    def load_ip_adapter(self, model_path):
        """Load weights for IP-Adapter from checkpoint."""
        if os.path.splitext(model_path)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        
        # Load weights for image projection model
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        
        # Explicitly move to the correct device after loading weights
        self.image_proj_model.to(self.device, dtype=self.dtype)
        
        # Load weights for IP-Adapter layers
        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])
        
        # Make sure attention processors are on the right device
        for processor in self.unet.attn_processors.values():
            processor.to(self.device, dtype=self.dtype)
    
    def load_controlnet(self, model_path):
        """Load weights for ControlNet from checkpoint."""
        if os.path.splitext(model_path)[-1] == ".safetensors":
            logger.warning("Please convert safetensors to bin format for controlnet loading")
            return
            
        state_dict = torch.load(model_path, map_location="cpu")
        self.controlnet.load_state_dict(state_dict["controlnet"])
    
    def set_ip_adapter_scale(self, scale):
        """Set the scale for IP-Adapter attention."""
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.scale = scale
    
    @torch.inference_mode()
    def _get_image_embeds(self, faceid_embeds, face_image, s_scale, shortcut, mix=False, mix_ratio=0.5):
        """Generate image embeddings for IP-Adapter."""
        if isinstance(face_image, PIL.Image.Image):
            face_image = [face_image]

        # Process images through CLIP
        clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]

        # Get unconditioned embeddings
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]

        # Mix embeddings if needed
        if mix:
            alpha = mix_ratio
            beta = 1 - alpha
            clip_image_embeds = (clip_image_embeds[0] * alpha + clip_image_embeds[1] * beta).unsqueeze(0)
            uncond_clip_image_embeds = (uncond_clip_image_embeds[0] * alpha + uncond_clip_image_embeds[1] * beta).unsqueeze(0)
        
        # Convert faceid_embeds to proper device and dtype
        faceid_embeds = faceid_embeds.to(self.device, dtype=self.dtype)
        
        # Project embeddings
        image_prompt_embeds = self.image_proj_model(faceid_embeds, clip_image_embeds, shortcut=shortcut, scale=s_scale)
        uncond_image_prompt_embeds = self.image_proj_model(
            torch.zeros_like(faceid_embeds), uncond_clip_image_embeds, shortcut=shortcut, scale=s_scale
        )
        
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        face_image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        faceid_embeds: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        scale: float = 1.0,  # IP-Adapter scale
        s_scale: float = 1.0,  # Face ID scale
        shortcut: bool = False,
        mix: bool = False,
        mix_ratio: float = 0.5,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Function invoked when calling the pipeline for generation.
        
        Args:
            prompt (`str` or `List[str]`): Text prompt for generation
            image (`PIL.Image.Image`): ControlNet input condition image
            face_image (`PIL.Image.Image`): Face image for IP-Adapter
            faceid_embeds (`torch.FloatTensor`): Pre-computed face embeddings
            height (`int`, *optional*): Height of the generated image
            width (`int`, *optional*): Width of the generated image
            num_inference_steps (`int`, *optional*, defaults to 50): Number of denoising steps
            guidance_scale (`float`, *optional*, defaults to 7.5): Guidance scale
            negative_prompt (`str` or `List[str]`, *optional*): Negative prompt
            num_images_per_prompt (`int`, *optional*, defaults to 1): Images per prompt
            scale (`float`, *optional*, defaults to 1.0): IP-Adapter scale
            s_scale (`float`, *optional*, defaults to 1.0): Face ID scale
            shortcut (`bool`, *optional*, defaults to False): Whether to use shortcut
            mix (`bool`, *optional*, defaults to False): Whether to mix face embeddings
            mix_ratio (`float`, *optional*, defaults to 0.5): Ratio for mixing (0.0=second image, 1.0=first image)
            seed (`int`, *optional*): Random seed for generation
            **kwargs: Additional arguments for the parent pipeline
        """
        # Set IP-Adapter scale
        self.set_ip_adapter_scale(scale)
        
        # Create a dictionary for parameters to pass to parent method
        call_kwargs = {
            "prompt": prompt,
            "image": image,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "num_images_per_prompt": num_images_per_prompt
        }
        
        # Set generator from seed if provided
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            call_kwargs["generator"] = generator
            
        # Process face embeddings if provided
        if faceid_embeds is not None and face_image is not None:
            # Get embeddings for IP-Adapter
            image_prompt_embeds, uncond_image_prompt_embeds = self._get_image_embeds(
                faceid_embeds, face_image, s_scale, shortcut, mix, mix_ratio
            )
            
            # Encode text prompt
            text_embeds = self.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                negative_prompt=negative_prompt,
            )
            prompt_embeds, negative_prompt_embeds = text_embeds
            
            # Duplicate embeddings for batch processing
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            
            # Concatenate text and image embeddings
            combined_prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            combined_negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            
            # Update call parameters with our embeddings, removing original prompts
            call_kwargs.update({
                "prompt_embeds": combined_prompt_embeds,
                "negative_prompt_embeds": combined_negative_prompt_embeds,
                "prompt": None,
                "negative_prompt": None
            })


        
        # Call the parent implementation with our parameters and any additional kwargs
        return super().__call__(**call_kwargs, **kwargs)