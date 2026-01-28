"""
Model handler for MedGemma 1.5 inference.
"""
import os
import torch
from PIL import Image
from typing import List, Optional
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText

# Load environment variables from .env file
load_dotenv()


def check_gpu_availability():
    """Check GPU availability and print diagnostics."""
    print("=" * 60)
    print("GPU Availability Check")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {device_name}")
        print(f"Current GPU: {torch.cuda.current_device()}")

        # Check for MIG (Multi-Instance GPU)
        gpu_name = torch.cuda.get_device_name(0)
        if "MIG" in gpu_name:
            print("Note: Running on MIG partition - using float32 for compatibility")
    else:
        print("CUDA is not available. Model will use CPU (slow).")
        print("\nTo use GPU, ensure you have:")
        print("1. NVIDIA GPU with CUDA support")
        print("2. CUDA toolkit installed")
        print("3. PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118")

    print("=" * 60)

    return cuda_available


class MedGemmaHandler:
    """Handler for MedGemma 1.5 model inference."""

    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it", device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
        self.use_float32 = False  # Flag for MIG compatibility

        # Check for local model path (useful for local development)
        local_model_path = os.path.join(os.path.dirname(__file__), "models", "medgemma-1.5-4b-it")
        if os.path.exists(local_model_path) and os.path.isfile(os.path.join(local_model_path, "config.json")):
            self.model_id = local_model_path
            print(f"Using local model from: {local_model_path}")
        else:
            print(f"Using model from Hugging Face Hub: {self.model_id}")

    def load_model(self):
        """Load the MedGemma 1.5 model and processor."""
        print(f"Loading MedGemma model: {self.model_id}")

        # Check GPU availability
        cuda_available = check_gpu_availability()

        # Determine device
        if self.device is None:
            if cuda_available:
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Using GPU: {gpu_name}")
                # Check for MIG partition - use float32 for compatibility
                if "MIG" in gpu_name:
                    self.use_float32 = True
                    print("MIG detected: Using float32 for CUBLAS compatibility")
            else:
                self.device = "cpu"
                self.use_float32 = True
                print("WARNING: Using CPU - this will be very slow!")
        else:
            print(f"Using device: {self.device}")

        # Get HF token from environment
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("Using Hugging Face token from .env file")
        else:
            print("Warning: No HF_TOKEN found in .env file")

        self.processor = AutoProcessor.from_pretrained(self.model_id, token=hf_token)

        # Load model with proper device configuration
        if self.device == "cuda" and torch.cuda.is_available():
            if self.use_float32:
                print("Loading model on GPU with float32 (MIG compatibility mode)...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    device_map="cuda",
                    token=hf_token,
                )
            else:
                print("Loading model on GPU with bfloat16...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                    token=hf_token,
                )
        else:
            print("Loading model on CPU (this may take a while)...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                token=hf_token,
            )

        print(f"Model loaded on device: {next(self.model.parameters()).device}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        print("Model loaded successfully!")

    def generate_report(
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 350,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """Generate a radiology report from medical images."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print(f"Processing {len(images)} images...")

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # Move to device - let the model handle dtype conversion
        inputs = inputs.to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        print(f"Input sequence length: {input_len}")

        with torch.inference_mode():
            if do_sample and temperature > 0:
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            else:
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            generation = generation[0][input_len:]

        report = self.processor.decode(generation, skip_special_tokens=True)

        # Clear GPU cache after inference
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("GPU cache cleared.")

        return report
