"""
Main Gradio application for MedGemma DICOM report drafting.
"""
# IMPORTANT: Import spaces FIRST before any CUDA-related packages (torch, transformers)
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False

import os
import traceback
from typing import Tuple, List

import gradio as gr
import torch

# Disable TF32 to avoid CUBLAS_STATUS_INVALID_VALUE errors with certain tensor shapes
# This forces cuBLAS to use more compatible computation paths
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from dicom_processor import process_dicom_study

# ============================================================================
# Model Loading - MUST be at module level for ZeroGPU compatibility
# ============================================================================
print("Loading MedGemma model at startup...")
MODEL_ID = os.getenv("MODEL_ID", "google/medgemma-1.5-4b-it")
HF_TOKEN = os.getenv("HF_TOKEN")

processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for better CUBLAS compatibility
    token=HF_TOKEN,
)
model.generation_config.do_sample = True
print(f"Model loaded: {MODEL_ID}")
print(f"Model device: {model.device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Store processed data for reuse
cached_data = {
    "zip_bytes": None,
    "images": None,
    "modality": None,
    "study_info": None
}


def process_dicom_file(
    file_path: str,
    max_slices_per_series: int,
    image_size: int,
    window_center: float,
    window_width: float,
    use_auto_window: bool
) -> Tuple[str, str, List[Image.Image]]:
    """Process uploaded DICOM ZIP file and return preview images."""
    global cached_data

    try:
        if file_path is None:
            return "No file uploaded", "", []

        with open(file_path, 'rb') as f:
            zip_bytes = f.read()

        # Use per-series sampling if max_slices_per_series > 0
        slices_per_series = max_slices_per_series if max_slices_per_series > 0 else None

        # Use auto window if checkbox is checked
        wc = None if use_auto_window else window_center
        ww = None if use_auto_window else window_width

        modality, images, study_info = process_dicom_study(
            zip_bytes,
            max_slices_per_series=slices_per_series,
            image_size=image_size,
            window_center=wc,
            window_width=ww
        )

        # Cache for later use in report generation
        cached_data["zip_bytes"] = zip_bytes
        cached_data["images"] = images
        cached_data["modality"] = modality
        cached_data["study_info"] = study_info

        max_per_series = study_info.get('MaxSlicesPerSeries', None)
        sampling_info = f"Max Slices Per Series: {max_per_series}" if max_per_series else "Sampling: Global (all series combined)"

        # Get window info
        default_wc = study_info.get('DefaultWindowCenter', 'N/A')
        default_ww = study_info.get('DefaultWindowWidth', 'N/A')
        window_info = f"Window: Auto (WC={default_wc}, WW={default_ww})" if use_auto_window else f"Window: Manual (WC={window_center}, WW={window_width})"

        # Estimate VRAM usage based on actual image size
        num_images = study_info.get('ProcessedImages', 0)
        img_size = study_info.get('ImageSize', 896)
        model_vram_gb = 8.0
        base_per_image_mb = 50
        size_factor = (img_size / 896) ** 2
        per_image_vram_mb = base_per_image_mb * size_factor
        images_vram_gb = (num_images * per_image_vram_mb) / 1024
        total_vram_gb = model_vram_gb + images_vram_gb

        info_text = f"""Study Information:

Modality: {study_info['Modality']}
Study Description: {study_info['StudyDescription']}
Study Date: {study_info['StudyDate']}
Patient ID: {study_info['PatientID']}

Series Count: {study_info.get('SeriesCount', 'N/A')}
Total Original Slices: {study_info.get('TotalOriginalSlices', 'N/A')}
{sampling_info}
Processed Images: {num_images}
Image Size: {img_size}x{img_size}
{window_info}

--- VRAM Estimate ---
Model: ~{model_vram_gb:.1f} GB
Images ({num_images} x {img_size}x{img_size}): ~{images_vram_gb:.1f} GB
Total Estimated: ~{total_vram_gb:.1f} GB
"""

        status = f"Processed {len(images)} images from {study_info['Modality']} study"

        return status, info_text, images

    except Exception as e:
        error_msg = f"Error processing DICOM: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg, "", []


def _generate_report_impl(
    file_path: str,
    max_slices_per_series: int,
    image_size: int,
    window_center: float,
    window_width: float,
    use_auto_window: bool,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
) -> str:
    """Generate radiology report using MedGemma."""
    global cached_data

    try:
        if file_path is None:
            return "Please upload a DICOM ZIP file first."

        # Check if we can use cached images
        use_cache = (
            cached_data["images"] is not None and
            cached_data["zip_bytes"] is not None
        )

        if use_cache:
            images = cached_data["images"]
            modality = cached_data["modality"]
        else:
            with open(file_path, 'rb') as f:
                zip_bytes = f.read()

            slices_per_series = max_slices_per_series if max_slices_per_series > 0 else None
            wc = None if use_auto_window else window_center
            ww = None if use_auto_window else window_width

            modality, images, study_info = process_dicom_study(
                zip_bytes,
                max_slices_per_series=slices_per_series,
                image_size=image_size,
                window_center=wc,
                window_width=ww
            )

        print(f"Processing {len(images)} images...")

        # Use custom prompt or default
        if not prompt.strip():
            prompt = f"You are a radiologist, please draft the full structured report for the following {modality} exam. Include the following sections: Technique, Findings, and Impression."

        # Build message content
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        # Process inputs
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device=model.device, dtype=torch.float16)

        input_len = inputs["input_ids"].shape[-1]
        print(f"Input sequence length: {input_len}")

        # Generate report
        with torch.inference_mode():
            if do_sample and temperature > 0:
                generation = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            else:
                generation = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            generation = generation[0][input_len:]

        report = processor.decode(generation, skip_special_tokens=True)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return report

    except Exception as e:
        error_msg = f"Error generating report: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


# Apply @spaces.GPU decorator if running on HuggingFace Spaces
if SPACES_AVAILABLE:
    @spaces.GPU(duration=120)
    def generate_report(
        file_path: str,
        max_slices_per_series: int,
        image_size: int,
        window_center: float,
        window_width: float,
        use_auto_window: bool,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> str:
        """Generate radiology report using MedGemma (GPU-accelerated on HF Spaces)."""
        return _generate_report_impl(
            file_path, max_slices_per_series, image_size,
            window_center, window_width, use_auto_window,
            prompt, max_tokens, temperature, top_p, top_k, do_sample
        )
else:
    def generate_report(
        file_path: str,
        max_slices_per_series: int,
        image_size: int,
        window_center: float,
        window_width: float,
        use_auto_window: bool,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> str:
        """Generate radiology report using MedGemma."""
        return _generate_report_impl(
            file_path, max_slices_per_series, image_size,
            window_center, window_width, use_auto_window,
            prompt, max_tokens, temperature, top_p, top_k, do_sample
        )


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(title="MedGemma 1.5 DICOM Report Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# MedGemma 1.5 DICOM Report Generator")
        gr.Markdown("Upload a ZIP file containing DICOM images to generate a structured radiology report.")

        with gr.Row():
            # Left column: Upload and settings
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload DICOM ZIP",
                    file_types=[".zip"],
                    type="filepath"
                )

                with gr.Accordion("Image Processing Settings", open=True):
                    max_slices_slider = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Max Slices Per Series",
                        info="0 = use all slices. Reduce to save VRAM."
                    )

                    image_size_slider = gr.Slider(
                        minimum=224,
                        maximum=1024,
                        value=512,
                        step=32,
                        label="Image Size",
                        info="Smaller = less VRAM, lower quality"
                    )

                    gr.Markdown("**Windowing (for CT/X-ray)**")
                    use_auto_window = gr.Checkbox(
                        label="Use Auto Window (from DICOM metadata)",
                        value=True
                    )
                    with gr.Row():
                        window_center_slider = gr.Slider(
                            minimum=-1000,
                            maximum=3000,
                            value=40,
                            step=10,
                            label="Window Center (WC)",
                            info="e.g., Brain=40, Lung=-600, Bone=400"
                        )
                        window_width_slider = gr.Slider(
                            minimum=1,
                            maximum=4000,
                            value=400,
                            step=10,
                            label="Window Width (WW)",
                            info="e.g., Brain=80, Lung=1500, Bone=1800"
                        )

                process_btn = gr.Button("Process & Preview", variant="primary", size="lg")

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )

                study_info_box = gr.Textbox(
                    label="Study Information & VRAM Estimate",
                    interactive=False,
                    lines=14
                )

            # Middle column: Image preview
            with gr.Column(scale=1):
                gr.Markdown("### Image Preview")
                gr.Markdown("*Preview of sampled slices that will be sent to the model*")

                image_gallery = gr.Gallery(
                    label="Sampled Slices",
                    show_label=False,
                    columns=4,
                    rows=3,
                    height=400,
                    object_fit="contain",
                    preview=True
                )

            # Right column: Generation settings and output
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    value="You are a radiologist, please draft the full structured report for this exam. Include: Technique, Findings, and Impression.",
                    info="Customize the prompt. Leave empty for default."
                )

                with gr.Accordion("Model Settings", open=False):
                    with gr.Row():
                        max_tokens_slider = gr.Slider(
                            minimum=50,
                            maximum=1000,
                            value=350,
                            step=10,
                            label="Max Tokens"
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                    with gr.Row():
                        top_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top P"
                        )
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Top K"
                        )
                    do_sample_checkbox = gr.Checkbox(
                        label="Enable Sampling",
                        value=True,
                        info="Uncheck for deterministic output"
                    )

                generate_btn = gr.Button("Generate Report", variant="primary", size="lg")

                report_output = gr.Textbox(
                    label="Generated Report",
                    interactive=False,
                    lines=18,
                    placeholder="Report will appear here..."
                )

        # Common window presets
        with gr.Accordion("Window Presets (click to apply)", open=False):
            gr.Markdown("**CT Presets:**")
            with gr.Row():
                brain_btn = gr.Button("Brain (40/80)", size="sm")
                subdural_btn = gr.Button("Subdural (75/215)", size="sm")
                stroke_btn = gr.Button("Stroke (32/8)", size="sm")
                lung_btn = gr.Button("Lung (-600/1500)", size="sm")
                mediastinum_btn = gr.Button("Mediastinum (50/350)", size="sm")
                bone_btn = gr.Button("Bone (400/1800)", size="sm")
                abdomen_btn = gr.Button("Abdomen (40/400)", size="sm")
                liver_btn = gr.Button("Liver (60/150)", size="sm")

        # Event handlers for presets
        brain_btn.click(lambda: (40, 80, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        subdural_btn.click(lambda: (75, 215, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        stroke_btn.click(lambda: (32, 8, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        lung_btn.click(lambda: (-600, 1500, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        mediastinum_btn.click(lambda: (50, 350, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        bone_btn.click(lambda: (400, 1800, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        abdomen_btn.click(lambda: (40, 400, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        liver_btn.click(lambda: (60, 150, False), outputs=[window_center_slider, window_width_slider, use_auto_window])

        # Main event handlers
        process_btn.click(
            fn=process_dicom_file,
            inputs=[
                file_input,
                max_slices_slider,
                image_size_slider,
                window_center_slider,
                window_width_slider,
                use_auto_window
            ],
            outputs=[status_output, study_info_box, image_gallery]
        )

        generate_btn.click(
            fn=generate_report,
            inputs=[
                file_input,
                max_slices_slider,
                image_size_slider,
                window_center_slider,
                window_width_slider,
                use_auto_window,
                prompt_input,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                do_sample_checkbox
            ],
            outputs=[report_output]
        )

        gr.Markdown("---")
        gr.Markdown("**Supported Modalities:** CT, MR, CR, DX | **Tip:** Use fewer slices and smaller image size to reduce VRAM usage")

    return demo


def main():
    """Main entry point."""
    print("Starting MedGemma 1.5 DICOM Report Generator...")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
