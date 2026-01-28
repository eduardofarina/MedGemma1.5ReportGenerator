---
title: MedGemma 1.5 Report Generator
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
license: mit
---

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

## Usage

1. Upload a ZIP file containing DICOM images

2. Adjust settings:
   - **Max Slices Per Series**: Reduce for less VRAM usage
   - **Image Size**: Smaller images use less VRAM
   - **Windowing**: Use presets or manual WC/WW for CT images

3. Click "Process & Preview" to see the sampled images and VRAM estimate

4. Click "Generate Report" to create the radiology report

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
- **Claude Opus** (Anthropic) for assistance in creating this demo
