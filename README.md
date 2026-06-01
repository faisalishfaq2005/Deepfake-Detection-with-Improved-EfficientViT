---

# Deepfake Detection with Improved EfficientViT

# Deepfake Detection - EfficientViT

[![Hugging Face Model](https://img.shields.io/badge/🤗_Hugging_Face-Model-blue)](https://huggingface.co/faisalishfaq2005/deepfake-detection-efficientnet-vit)

Official GitHub repository for the model hosted on Hugging Face.

## Model Link
→ **[faisalishfaq2005/deepfake-detection-efficientnet-vit](https://huggingface.co/faisalishfaq2005/deepfake-detection-efficientnet-vit)**



## Model Architecture

<img width="1007" height="579" alt="deepfake_architecture_diagram_edited drawio (1)" src="https://github.com/user-attachments/assets/8a96c427-b39b-47d5-b824-705ea1b93e61" />

## Inference Pipeline

<img width="750" height="434" alt="inferencepipeline drawio" src="https://github.com/user-attachments/assets/509fb69d-5cc5-43e5-8ded-d6aafc6eceb2" />




This repository contains a **PyTorch model for deepfake detection** based on an improved **EfficientViT** architecture, trained on video data.  

The model predicts whether a video is **real (0)** or **fake (1)** using both visual information and temporal cues.

---

## 🧩 Model Description

**Architecture:** Improved EfficientViT  
**Backbone:** EfficientNet-B0 for feature extraction  
**Head:** Transformer-based temporal modeling with classification head  
**Input:** Video frames (224×224 RGB images)  
**Output:** Binary label (0=Real, 1=Fake) and frame-level probabilities  

**Key Features:**

- Extracts faces from frames using MTCNN  
- Supports inference on raw video files  
- Provides frame-level probabilities for fine-grained analysis  

---

## 📁 Repository Structure

```
deepfake-efficientvit/
│
├── model.py                  # ImprovedEfficientViT class
├── inference.py              # Functions to run inference on videos
├── model.pth  # Trained weights
├── config.json               # Optional model metadata
├── requirements.txt          # Required packages
├── README.md

```

## ⚡ Installation
git clone https://huggingface.co/faisalishfaq2005/deepfake-detection-efficientnet-vit

cd deepfake-detection-efficientnet-vit

pip install -r requirements.txt

## 🚀 Usage
# 1.Programmatic Inference

```python

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
from model import ImprovedEfficientViT
from inference import predict_vedio 

# 1️⃣ Download the checkpoint from Hugging Face
checkpoint_path = hf_hub_download(
    repo_id="faisalishfaq2005/deepfake-detection-efficientnet-vit",  
    filename="model.safetensors"
)

# 2️⃣ Load the model weights safely
state_dict = load_file(checkpoint_path, device="cpu")
model = ImprovedEfficientViT()
model.load_state_dict(state_dict)
model.eval()

# 4️⃣ Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 3️⃣ Run inference on a video
video_path = "sample_video.mp4"
result = predict_vedio(video_path, model)
print(result)
# Example Output: {'class': 1}

```
# 2. Manual Download

Go to the Hugging Face model page

Download:

model.pth

model.py

inference.py

Place them in the same folder locally.

Install requirements and run predict_video().

## 📄 License

This model is released under the MIT License.
You are free to use, modify, and distribute it, with attribution.

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@inproceedings{faisalishfaq2025efficientvit,
  title={Deepfake Detection with Efficientnet and ViT},
  author={Faisal Ishfaq},
  year={2025}
}
```


