# I-JEPA: Image-based Joint-Embedding Predictive Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains an implementation of the Image-based Joint-Embedding Predictive Architecture (I-JEPA) as proposed by Yann LeCun. I-JEPA is a self-supervised learning framework that learns visual representations by predicting masked regions of images using a context-target prediction mechanism.

## Architecture Overview

I-JEPA consists of three main components:

1. **Context Encoder**: A Vision Transformer (ViT) that processes visible regions of the input image
2. **Target Encoder**: A momentum-updated copy of the context encoder that processes target regions
3. **Predictor**: A lightweight transformer that predicts target representations from context representations

### Key Features

- **Masking Strategy**: Implements a sophisticated masking mechanism with:
  - Multiple target blocks (default: 4)
  - Configurable scale ranges for context (0.85-1.0) and target (0.15-0.2) blocks
  - Adjustable aspect ratios for target blocks (0.75-1.5)

- **Model Variants**:
  - Base: ViT-B/16 (768 dim, 12 layers, 12 heads)
  - Large: ViT-L/16 (1024 dim, 24 layers, 16 heads)
  - Huge: ViT-H/14 (1280 dim, 32 layers, 16 heads)

## Installation

```bash
# Clone the repository
git clone https://github.com/The-Swarm-Corporation/JEPA.git
cd JEPA

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import torch
from jepa import create_ijepa_model, MaskGenerator

# Create model
model = create_ijepa_model(
    model_size='base',
    img_size=224,
    patch_size=16
)

# Prepare input
batch_size = 4
images = torch.randn(batch_size, 3, 224, 224)

# Generate masks
mask_generator = MaskGenerator(grid_size=14)  # 224/16 = 14
context_mask, target_masks = mask_generator.generate_masks(
    batch_size=batch_size,
    device=images.device
)

# Forward pass
outputs = model(images, context_mask, target_masks)
```

### Model Configuration

```python
model = create_ijepa_model(
    model_size='base',          # 'base', 'large', or 'huge'
    img_size=224,              # Input image size
    patch_size=16,             # Patch size for tokenization
    predictor_dim=384,         # Predictor's internal dimension
    momentum=0.996             # EMA momentum for target encoder
)
```

## Model Architecture Details

### Vision Transformer (Context/Target Encoder)

- **Patch Embedding**: Converts image patches to embeddings
- **Positional Encoding**: 2D sinusoidal position embeddings
- **Transformer Blocks**: Self-attention and MLP layers
- **Layer Normalization**: Applied before attention and MLP

### Predictor

- **Input Projection**: Projects context representations to predictor dimension
- **Mask Tokens**: Learnable tokens for target positions
- **Transformer Blocks**: Cross-attention between context and target
- **Output Projection**: Projects predictions to target dimension

## Training Methodology

1. **Masking Process**:
   - Generate context block (85-100% of image)
   - Generate 4 target blocks (15-20% each)
   - Ensure no overlap between context and target regions

2. **Forward Pass**:
   - Context encoder processes visible regions
   - Target encoder processes full image (momentum-updated)
   - Predictor estimates target representations from context

3. **Loss Computation**:
   - MSE loss between predicted and actual target representations
   - Average loss across all target blocks

## Requirements

- Python 3.10+
- PyTorch 2.0+
- einops
- loguru
- matplotlib
- numpy


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Implementation inspired by Yann LeCun's work on self-supervised learning
- Built with the Swarms Framework

## Contact

- Twitter: [@kyegomez](https://twitter.com/kyegomezb)
- Email: kye@swarms.world
