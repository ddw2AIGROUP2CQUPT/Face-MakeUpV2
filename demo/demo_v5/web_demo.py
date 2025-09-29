import gradio as gr
import torch
import os
import sys
import tempfile
import numpy as np

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel
from models.FaceMakeUp2.FaceMakeUpPipeline import FaceMakeUpPipeline
from utils.face_extraction_v5 import FaceExtractor, ShadingGenerator

class FaceMakeUpDemo:
    def __init__(self):
        self.pipe = None
        self.face_extractor = None
        self.shading_generator = None
        self.device = "cuda:6" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        
    def initialize_models(self):
        """Automatically initialize models"""
        if self.model_loaded:
            return

        # Fixed model path configuration
        base_model_path = "pretrained/SG161222/Realistic_Vision_V4.0_noVAE"
        vae_model_path = "stabilityai/sd-vae-ft-mse"
        model_path = "checkpoints/train_mask_4face_short_caption_v5_id/checkpoint-680000/model.bin"
        image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        lora_rank = 128
        
        print("🚀 Automatically initializing models...")
        print(f"📍 Using device: {self.device}")
        print(f"📂 Model path: {model_path}")
        
        try:
            # Initialize scheduler
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )
            
            print("📦 Loading VAE model...")
            vae = AutoencoderKL.from_pretrained(vae_model_path)
            
            print("📦 Loading UNet model...")
            unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
            
            print("📦 Creating ControlNet...")
            controlnet = ControlNetModel.from_unet(unet)
            
            print("📦 Building Pipeline...")
            self.pipe = FaceMakeUpPipeline.from_pretrained(
                base_model_path,
                vae=vae,
                controlnet=controlnet,
                scheduler=noise_scheduler,
                torch_dtype=torch.float32,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                image_encoder_path=image_encoder_path,
                lora_rank=lora_rank
            )
            
            print(f"🔄 Moving model to {self.device}...")
            self.pipe = self.pipe.to(self.device)
            
            print("📦 Loading IP-Adapter...")
            self.pipe.load_ip_adapter(model_path)
            
            print("📦 Loading ControlNet weights...")
            self.pipe.load_controlnet(model_path)
            
            print("👤 Initializing face extractor...")
            self.face_extractor = FaceExtractor(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            
            print("🎨 Initializing shading generator...")
            DECA_DATA_FOLDER = "models/Relightable-Portrait-Animation/src/decalib/data/"
            self.shading_generator = ShadingGenerator(deca_data_path=DECA_DATA_FOLDER)
            
            self.model_loaded = True
            print("✅ Model initialization complete! System is ready.")
            
        except Exception as e:
            print(f"❌ Model initialization failed: {str(e)}")
            raise e
    
    def generate_makeup(self,
                          identity_image,
                          shading_ref_image,
                          prompt,
                          negative_prompt,  # New parameter
                          num_images,
                          scale=0.7,
                          s_scale=0.7,
                          controlnet_conditioning_scale=0.5,
                          seed=42,
                          shadingsize_scale_factor=5.0,
                          imgsize_scale_factor=3.0,
                          num_inference_steps=50,
                          width=512,
                          height=512,
                          progress=gr.Progress()):
        
        if not self.model_loaded:
            return None, None, None, "❌ Model not loaded yet, please try again later!"

        if identity_image is None:
            return None, None, None, "❌ Please upload an identity image!"

        if not prompt.strip():
            return None, None, None, "❌ Please enter a prompt!"
        
        temp_identity_path = None
        temp_shading_path = None
        try:
            # Determine which image to use as lighting reference
            shading_source_image = shading_ref_image if shading_ref_image is not None else identity_image
            
            # 1. Process identity image
            progress(0.1, desc="Processing...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                identity_image.save(tmp_file.name, quality=95)
                temp_identity_path = tmp_file.name
            
            face_image, faceid_embeds = self.face_extractor.extract_features(
                temp_identity_path, imgsize_scale_factor=imgsize_scale_factor
            )
            
            # 2. Process pose/lighting image
            progress(0.3, desc="Processing...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                shading_source_image.save(tmp_file.name, quality=95)
                temp_shading_path = tmp_file.name
                
            face_image_shading, _ = self.face_extractor.extract_features(
                temp_shading_path, imgsize_scale_factor=shadingsize_scale_factor
            )
            shading_image = self.shading_generator.generate_from_image(face_image=face_image_shading)
            
            if shading_image is None or face_image is None or faceid_embeds is None:
                return None, None, None, "❌ Unable to extract facial features from image, please ensure the image contains a clear face."
            
            progress(0.6, desc="Preparing generation parameters...")
            if not isinstance(faceid_embeds, torch.Tensor):
                faceid_embeds = torch.tensor(faceid_embeds, dtype=torch.float32)
            
            # Modified: Use user input prompt, use default if empty
            positive_prompt = prompt.strip() if prompt.strip() else "best quality"
            if not positive_prompt.startswith("best quality"):
                positive_prompt = "best quality " + positive_prompt
            
            # Modified: Use user input negative prompt, use default if empty
            final_negative_prompt = negative_prompt.strip() if negative_prompt.strip() else (
                "black image, Easy Negative, worst quality, low quality, lowers, monochrome, grayscales, "
                "skin spots, acnes, skin blemishes, age spot, 6 more fingers on one hand, deformity, bad legs"
            )
            
            all_images = []
            num_images_int = int(num_images)
            
            for i in range(num_images_int):
                progress(0.7 + (i / num_images_int) * 0.3, desc=f"Generating image {i + 1}/{num_images_int}...")
                current_seed = seed + i if seed >= 0 else -1
                generator = torch.Generator(device=self.device).manual_seed(current_seed) if current_seed >= 0 else torch.Generator(device=self.device).seed()
                
                output = self.pipe(
                    prompt=positive_prompt,
                    negative_prompt=final_negative_prompt,  # Use user-defined negative prompt
                    image=shading_image,
                    face_image=face_image,
                    faceid_embeds=faceid_embeds,
                    scale=scale,
                    s_scale=s_scale,
                    shortcut=True,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                )
                all_images.append(output.images[0])
            
            progress(1.0, desc="Generation complete!")
            
            success_message = f"✅ Successfully generated {len(all_images)} images! Using initial seed: {seed}"
            
            # Return different formats based on number of images
            if len(all_images) == 1:
                return all_images[0], all_images, shading_image, success_message
            else:
                return None, all_images, shading_image, success_message
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, None, f"❌ Error occurred during generation: {str(e)}"
        
        finally:
            # Clean up all temporary files
            if temp_identity_path and os.path.exists(temp_identity_path):
                os.unlink(temp_identity_path)
            if temp_shading_path and os.path.exists(temp_shading_path):
                os.unlink(temp_shading_path)

# Global instance
demo_instance = FaceMakeUpDemo()

def create_interface():
    default_seed = np.random.randint(0, 2147483647)
    
    # Default negative prompt
    default_negative_prompt = (
        "black image, Easy Negative, worst quality, low quality, lowers, monochrome, grayscales, "
        "skin spots, acnes, skin blemishes, age spot, 6 more fingers on one hand, deformity, bad legs"
    )
    
    # Create simple custom theme
    custom_theme = gr.themes.Base(
        primary_hue="slate",  # Use gray tones
        secondary_hue="slate",
        neutral_hue="slate",
        font=("Inter", "ui-sans-serif", "system-ui"),
        text_size="lg",
    ).set(
        # Set main colors to deeper gray tones
        button_primary_background_fill="#495057",
        button_primary_background_fill_hover="#343a40",
        button_primary_border_color="#495057",
        button_primary_text_color="white",

        # Set input box style
        input_background_fill="#f1f3f4",
        input_border_color="#ced4da",
        input_border_width="1px",

        # Set label color - deeper gray
        block_label_text_color="#2c3e50",
        block_title_text_color="#1a202c",

        # Set background color
        background_fill_primary="#ffffff",
        background_fill_secondary="#f1f3f4",
    )

    custom_css = """
    .custom-textbox label {
        font-family: 'Arial', sans-serif !important;
        font-size: 33px !important;
        font-weight: bold !important;
    }
    .custom-textbox input, .custom-textbox textarea {
        font-family: 'Arial', sans-serif !important;
        font-size: 45px !important;
        color: red !important;
        font-weight: bold !important;
    }
    .custom-textbox input::placeholder, .custom-textbox textarea::placeholder {
        font-family: 'Arial', sans-serif !important;
        font-size: 45px !important;
        font-style: italic !important;
        color: #666 !important;
    }

    /* 优化图片显示，填充区域但保持比例 */
    .gradio-container img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
    }

    /* 单张图片显示优化 */
    .gradio-image img {
        width: 100% !important;
        height: auto !important;
        max-height: 800px !important;
        object-fit: contain !important;
        object-position: center !important;
    }

    /* 画廊显示优化 */
    .gradio-gallery .grid-wrap {
        gap: 8px !important;
    }

    .gradio-gallery .thumbnail-item img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        object-position: center !important;
    }
    """

    with gr.Blocks(title="Face-MakeUp AI", theme=custom_theme, css=custom_css) as demo:
        
        model_status = gr.HTML("""
            <div style="text-align: center; padding: 20px; background: #f8f9fa; color: #2c3e50; border: 1px solid #e9ecef; border-radius: 10px; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2.8em; font-weight: 600;">🎨 Face-MakeUpV2</h1>
            </div>
        """)
        
        with gr.Row():
            # Left section: Reference and Target columns
            with gr.Column(scale=1, elem_classes="left-column"):
                with gr.Column(elem_classes="content-area"):
                    with gr.Row():
                        # Reference column (extended area)
                        with gr.Column(scale=1):
                            gr.Markdown('<div style="background: #f8f9fa; padding: 12px; border-left: 4px solid #6c757d; border-radius: 6px; margin-bottom: 15px;"><h3 style="margin: 0; color: #495057; font-size: 1.95em;">👤 Reference</h3></div>')
                            identity_image = gr.Image(sources=['upload'], show_label=False, type="pil", height=430)

                        # Target column (two image blocks vertically stacked)
                        with gr.Column(scale=1):
                            gr.Markdown('<div style="background: #f8f9fa; padding: 12px; border-left: 4px solid #6c757d; border-radius: 6px; margin-bottom: 15px;"><h3 style="margin: 0; color: #495057; font-size: 1.95em;">💡 Target</h3></div>')
                            shading_ref_image = gr.Image(sources=['upload'], label="Target Image(Optional)", type="pil", height=180)
                            shading_visualization = gr.Image(label="Target Shading Map", height=230, interactive=False)

                    # Prompt area below the first two columns
                    gr.Markdown('<div style="background: #f8f9fa; padding: 12px; border-left: 4px solid #6c757d; border-radius: 6px; margin-bottom: 15px;"><h3 style="margin: 0; color: #495057; font-size: 1.95em;">⚙️ Text Prompt </h3></div>')

                    prompt = gr.Textbox(
                        show_label=False,
                        placeholder="For example: elegant red lipstick, smoky eye makeup",
                        lines=2,
                        value="",
                        elem_classes="custom-textbox"
                    )

                    with gr.Accordion("🔧 Advanced Parameter Settings", open=False):
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Describe effects you don't want, for example: blurry, bad quality, distorted face",
                            lines=2,
                            value=default_negative_prompt,
                            info="Used to specify features you don't want to appear, leave blank to use default"
                        )
                        with gr.Row():
                            scale = gr.Slider(minimum=0.1, maximum=2.0, value=0.5, step=0.1, label="IP-Adapter Strength", info="Control style transfer strength")
                            s_scale = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Facial ID Preservation", info="Control original facial feature preservation")

                        with gr.Row():
                            controlnet_conditioning_scale = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="ControlNet Strength", info="Control structure preservation strength")
                            num_inference_steps = gr.Slider(minimum=20, maximum=100, value=50, step=5, label="Generation Steps", info="More steps = higher quality, slower speed")


                        with gr.Row():
                            num_images = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Number of Images", info="Choose how many images to generate at once")

                        with gr.Row():
                            seed = gr.Number(label="Random Seed", value=default_seed, precision=0, info="Same seed produces same result")
                            random_seed_btn = gr.Button("🎲 Random", size="sm")

                        with gr.Row():
                            imgsize_scale_factor = gr.Slider(minimum=1.0, maximum=5.0, value=3.0, step=0.1, label="Image Scale", info="Face region scaling factor")
                            shadingsize_scale_factor = gr.Slider(minimum=2.0, maximum=8.0, value=5.0, step=0.1, label="Shading Scale", info="Shading generation region scaling factor")

                        with gr.Row():
                            width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width", info="Generated image width")
                            height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height", info="Generated image height")

                with gr.Column(elem_classes="button-area"):
                    # gr.HTML('<div style="height: 280px;"></div>')  # Fixed spacing to align Generate button with gallery bottom
                    generate_btn = gr.Button("🎨 Generate", variant="primary", size="lg")

                    status_text = gr.Textbox(label="Status Information", interactive=False, lines=2)
                    clear_btn = gr.Button("🗑️ Clear Results", variant="secondary")

            # Right section: Generated Images (remains on the right side)
            with gr.Column(scale=1):
                gr.Markdown('<div style="background: #f8f9fa; padding: 12px; border-left: 4px solid #6c757d; border-radius: 6px; margin-bottom: 15px;"><h3 style="margin: 0; color: #495057; font-size: 1.95em;">🎨 The Generated Images</h3></div>')

                # Single image display (visible when generating only 1 image)
                output_single = gr.Image(show_label=False, height=800, interactive=False, visible=False,
                                       container=True, show_download_button=False, show_share_button=False, sources=[])

                # Gallery display (visible when generating multiple images)
                output_gallery = gr.Gallery(show_label=False, height=800, object_fit="cover", columns=2, preview=True, visible=True,
                                          container=True, show_download_button=False, show_share_button=False)
        
        
        def update_seed():
            return np.random.randint(0, 2147483647)


        def handle_generation_output(single_image, gallery_images, shading_image, status):
            """Handle the output display based on number of images generated"""
            if single_image is not None:
                # Single image mode - show single image, hide gallery
                return (
                    gr.update(value=single_image, visible=True),  # output_single
                    gr.update(value=gallery_images, visible=False),  # output_gallery
                    shading_image,  # shading_visualization
                    status  # status_text
                )
            elif gallery_images is not None:
                # Multiple images mode - hide single image, show gallery
                # Adjust columns based on number of images for better display
                num_images = len(gallery_images) if gallery_images else 1
                columns = 1 if num_images <= 2 else 2
                return (
                    gr.update(visible=False),  # output_single
                    gr.update(value=gallery_images, visible=True, columns=columns),  # output_gallery
                    shading_image,  # shading_visualization
                    status  # status_text
                )
            else:
                # Error case - hide both
                return (
                    gr.update(visible=False),  # output_single
                    gr.update(visible=False),  # output_gallery
                    shading_image,  # shading_visualization
                    status  # status_text
                )


        def clear_outputs():
            return (
                gr.update(value=None, visible=False),  # output_single
                gr.update(value=None, visible=True),   # output_gallery (default visible)
                None,  # shading_visualization
                ""     # status_text
            )
        
        
        # Event binding
        random_seed_btn.click(fn=update_seed, outputs=seed)
        
        # Modified: Add negative_prompt, width, height to input parameters
        generate_btn.click(
            fn=demo_instance.generate_makeup,
            inputs=[
                identity_image, shading_ref_image, prompt, negative_prompt, num_images, scale, s_scale,
                controlnet_conditioning_scale, seed, shadingsize_scale_factor,
                imgsize_scale_factor, num_inference_steps, width, height
            ],
            outputs=[output_single, output_gallery, shading_visualization, status_text]
        ).then(
            fn=handle_generation_output,
            inputs=[output_single, output_gallery, shading_visualization, status_text],
            outputs=[output_single, output_gallery, shading_visualization, status_text]
        )

        clear_btn.click(
            fn=clear_outputs,
            outputs=[output_single, output_gallery, shading_visualization, status_text]
        )
    
    return demo

def main():
    """Main startup function"""
    print("🚀 Starting Face-MakeUpV2...")
    print("=" * 50)
    
    model_path = "checkpoints/train_mask_4face_short_caption_v5_id/checkpoint-680000/model.bin"
    if not os.path.exists(model_path):
        print(f"❌ Warning: Model file does not exist: {model_path}")
        print("Please ensure the model file path is correct!")
    
    try:
        demo_instance.initialize_models()
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        print("Program will continue running, but functionality may be limited")
    
    demo = create_interface()
    
    print("=" * 50)
    print("🌐 Web interface started!")
    print("📍 Local access address: http://localhost:9880")
    print("🛑 Press Ctrl+C to stop service")
    print("=" * 50)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=9880,
        share=False,
        debug=False,
        show_error=True, 
        quiet=False
    )

if __name__ == "__main__":
    main()