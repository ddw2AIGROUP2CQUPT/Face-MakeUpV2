# Face-MakeUpV2 🎨

An advanced AI-powered face makeup generation system based on IP-Adapter and ControlNet.

## 🌟 Features

- **Face-ID Preservation**: Maintains facial identity while applying makeup
- **Style Transfer**: Apply various makeup styles from reference images
- **Attention Mask Guidance**: Precise control over makeup application areas
- **ControlNet Integration**: Structural guidance for consistent results
- **Web Interface**: User-friendly Gradio-based demo

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/Face-MakeUpV2.git
cd Face-MakeUpV2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pretrained models (see Model Setup section)

## 📦 Model Setup

### Required Models:
- Base diffusion model (e.g., Realistic_Vision_V4.0)
- CLIP vision encoder
- Face recognition model
- Trained IP-Adapter weights

Please refer to the documentation for detailed model setup instructions.

## 🎯 Usage

### Web Demo
```bash
python demo/demo_v5/web_demo.py
```

### Training
```bash
bash train/mask/lora/train_v5.sh
```

## 📁 Project Structure

```
Face-MakeUpV2/
├── train/              # Training scripts
├── models/             # Model architectures
├── dataset/            # Dataset processing
├── utils/              # Utility functions
├── demo/               # Demo applications
└── examples/           # Usage examples
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Citation

If you use this code in your research, please cite:

```bibtex
@article{facemakeupv2,
  title={Face-MakeUpV2: Advanced AI Face Makeup Generation},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 🙏 Acknowledgments

- IP-Adapter team for the foundational work
- Diffusers library for the pipeline implementation
- ControlNet for structural guidance
