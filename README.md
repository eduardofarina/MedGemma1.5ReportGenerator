# MedGemma 1.5 DICOM Report Generator

A Gradio-based web application that uses Google's MedGemma 1.5 model to automatically generate structured radiology reports from DICOM medical images.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **DICOM Processing**: Upload ZIP files containing DICOM images from CT, MR, CR, or DX studies
- **Smart Sampling**: Configurable slice sampling per series to manage GPU memory
- **DICOM Windowing**: Auto or manual window/level controls with CT presets (Brain, Lung, Bone, etc.)
- **Image Preview**: Built-in gallery to visualize sampled slices before inference
- **VRAM Estimation**: Real-time estimation of GPU memory usage based on settings
- **Configurable Generation**: Adjustable temperature, top-p, top-k, and max tokens
- **Custom Prompts**: Editable prompts for tailored report generation

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended: 12GB+ VRAM)
- Hugging Face account with access to [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/eduardofarina/MedGemma1.5ReportGenerator.git
cd MedGemma1.5ReportGenerator
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate medgemma
```

3. Create a `.env` file with your Hugging Face token:
```bash
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

4. (Optional) Pre-download the model:
```bash
python download_model.py
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser to `http://localhost:7860`

3. Upload a ZIP file containing DICOM images

4. Adjust settings:
   - **Max Slices Per Series**: Reduce for less VRAM usage
   - **Image Size**: Smaller images use less VRAM
   - **Windowing**: Use presets or manual WC/WW for CT images

5. Click "Process & Preview" to see the sampled images and VRAM estimate

6. Click "Generate Report" to create the radiology report

## Window Presets

| Preset | Window Center | Window Width | Use Case |
|--------|--------------|--------------|----------|
| Brain | 40 | 80 | Brain parenchyma |
| Subdural | 75 | 215 | Subdural hematoma |
| Stroke | 32 | 8 | Acute stroke |
| Lung | -600 | 1500 | Lung parenchyma |
| Mediastinum | 50 | 350 | Mediastinal structures |
| Bone | 400 | 1800 | Bone windows |
| Abdomen | 40 | 400 | Abdominal soft tissue |
| Liver | 60 | 150 | Liver lesions |

## Project Structure

```
MedGemma1.5ReportGenerator/
├── app.py                 # Main Gradio application
├── model_handler.py       # MedGemma model loading and inference
├── dicom_processor.py     # DICOM processing utilities
├── download_model.py      # Model download script
├── environment.yml        # Conda environment specification
├── requirements.txt       # Pip requirements
├── .env                   # Hugging Face token (not tracked)
└── models/                # Local model cache (not tracked)
```

## Tips for Low VRAM

- Use **Max Slices Per Series = 5-10** instead of all slices
- Reduce **Image Size** to 256-384 pixels
- Process one series at a time for very large studies

## Disclaimer

This tool is for research and educational purposes only. It is NOT intended for clinical use or medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## License

MIT License

## Acknowledgments

- [Google MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) for the medical vision-language model
- [Gradio](https://gradio.app/) for the web interface framework
- [PyDICOM](https://pydicom.github.io/) for DICOM file processing
- **Claude Opus** (Anthropic) for assistance in creating this demo in under an hour
