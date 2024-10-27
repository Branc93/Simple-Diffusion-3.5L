# Import statements at the top
import torch
import torchvision
torchvision.disable_beta_transforms_warning()

from diffusers import StableDiffusion3Pipeline
import gradio as gr
import time
import webbrowser
from threading import Timer
import gc
import os
from typing import Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from PIL import Image

@dataclass
class GenerationResult:
    """Dataclass to store generation results and metadata"""
    image: Optional[Image.Image] = None
    settings_info: str = ""
    error: str = ""
    success: bool = True

class MemoryManager:
    """Handles memory management operations"""
    @staticmethod
    def setup_memory_config():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def cleanup():
        """Aggressive cleanup of GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()

class CustomSD3Pipeline:
    """Handles the Stable Diffusion pipeline operations"""
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_pipeline()

    def setup_pipeline(self):
        """Initialize the pipeline with optimal settings"""
        print(f"Loading model on {self.device.upper()}...")
        try:
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                "ariG23498/sd-3.5-merged",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                text_encoder_3=None,  # Disable T5 encoder to save VRAM
                tokenizer_3=None,     # Disable T5 tokenizer
                low_cpu_mem_usage=True
            )
            
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing(slice_size="max")
                self.pipeline.enable_model_cpu_offload()
                
                if hasattr(self.pipeline, 'vae'):
                    self.pipeline.vae.enable_tiling()
            else:
                self.pipeline = self.pipeline.to(self.device)
                
            print(f"Model loaded on {self.device.upper()} successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def generate(self, prompt: str, height: int, width: int, num_steps: int = 6,
                guidance_scale: float = 1.0, seed: int = 0, negative_prompt: str = None) -> GenerationResult:
        """Generate an image with optimized parameters"""
        MemoryManager.cleanup()
        start_time = time.time()
        
        try:
            if not prompt:
                return GenerationResult(success=False, error="Prompt cannot be empty")
            if height > 2048 or width > 2048:
                return GenerationResult(success=False, error="Maximum dimension size is 2048x2048")

            generator = torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed(seed)
            
            with torch.inference_mode():
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    generator=generator
                )
                
            image = output.images[0]
            
            end_time = time.time()
            generation_time = round(end_time - start_time, 2)
            
            vram_usage = ""
            if self.device == "cuda":
                vram_used = torch.cuda.max_memory_allocated() / 1024**3
                vram_usage = f"\nVRAM Usage: {vram_used:.2f}GB"
                
            settings_info = (
                f"Size: {width}x{height}\n"
                f"Steps: {num_steps}\n"
                f"CFG: {guidance_scale}\n"
                f"Seed: {seed}\n"
                f"Generation Time: {generation_time}s"
                f"{vram_usage}"
            )
            
            return GenerationResult(image=image, settings_info=settings_info)
            
        except torch.cuda.OutOfMemoryError:
            MemoryManager.cleanup()
            return GenerationResult(
                success=False,
                error="Out of memory. Try reducing the image size or number of steps."
            )
        except Exception as e:
            MemoryManager.cleanup()
            return GenerationResult(success=False, error=f"Error: {str(e)}")

class GradioInterface:
    """Handles the Gradio interface setup and operations"""
    def __init__(self):
        self.pipeline = CustomSD3Pipeline()
        self.demo = None
        self.setup_interface()

    def setup_interface(self):
        """Setup the Gradio interface with improved styling and features"""
        with gr.Blocks(theme=gr.themes.Soft()) as self.demo:
            gr.HTML("""
                <div style='text-align: center; margin-bottom: 1rem'>
                    <h1 style='margin-bottom: 0.5rem'>Simple-Diffusion-3.5L</h1>
                    <p style='font-style: italic; color: #666; font-size: 0.9rem'>
                        Credit to https://huggingface.co/ariG23498/sd-3.5-merged<br>
                        Thank you ariG23498 and stabilityAI
                    </p>
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to create...",
                        lines=3
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What you don't want to see in the image...",
                        lines=2
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            width = gr.Slider(
                                label="Width",
                                minimum=256,
                                maximum=2048,
                                step=64,
                                value=512
                            )
                            height = gr.Slider(
                                label="Height",
                                minimum=256,
                                maximum=2048,
                                step=64,
                                value=512
                            )
                        
                        with gr.Column():
                            steps = gr.Slider(
                                label="Steps",
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=6
                            )
                            guidance = gr.Slider(
                                label="CFG Scale",
                                minimum=1,
                                maximum=20,
                                step=0.5,
                                value=1
                            )
                    
                    seed = gr.Number(
                        label="Seed (0 for random)",
                        value=0,
                        precision=0
                    )
                    generate_btn = gr.Button(
                        "Generate",
                        variant="primary"
                    )
                    
                with gr.Column(scale=2):
                    result_image = gr.Image(
                        label="Generated Image",
                        type="pil"
                    )
                    settings_info = gr.Textbox(
                        label="Generation Info",
                        interactive=False,
                        lines=6
                    )

            def generate_wrapper(prompt, negative_prompt, width, height, steps, guidance, seed):
                result = self.pipeline.generate(
                    prompt=prompt,
                    height=int(height),
                    width=int(width),
                    num_steps=int(steps),
                    guidance_scale=float(guidance),
                    seed=int(seed) if seed != 0 else int(time.time()),
                    negative_prompt=negative_prompt if negative_prompt else None
                )
                
                if not result.success:
                    return None, result.error
                
                return result.image, result.settings_info

            generate_btn.click(
                fn=generate_wrapper,
                inputs=[
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    steps,
                    guidance,
                    seed
                ],
                outputs=[
                    result_image,
                    settings_info
                ]
            )

        self.demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            inbrowser=True
        )

if __name__ == "__main__":
    try:
        interface = GradioInterface()
    except Exception as e:
        print(f"Error initializing interface: {str(e)}")
        input("Press Enter to exit...")
