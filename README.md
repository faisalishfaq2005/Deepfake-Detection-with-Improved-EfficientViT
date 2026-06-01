---

# Deepfake Detection with Improved EfficientViT


[![Hugging Face Model](https://img.shields.io/badge/🤗_Hugging_Face-Model-blue)](https://huggingface.co/faisalishfaq2005/deepfake-detection-efficientnet-vit)

Official GitHub repository for the model hosted on Hugging Face.

## Model Link
→ **[faisalishfaq2005/deepfake-detection-efficientnet-vit](https://huggingface.co/faisalishfaq2005/deepfake-detection-efficientnet-vit)**


## Data Preprocessing

In this study, we utilized the free version of FaceForen-
sics++ dataset and the Celeb-DF dataset available
on Kaggle. From these datasets, we selected a total of 1,580
videos, with an equal distribution of 790 real and 790 fake
videos.
To prepare the data for training, we first calculated the
duration of each video to ensure consistent frame extraction.
We then extracted 20 evenly spaced frames from each video,
allowing us to capture representative visual information while
avoiding redundancy. For each extracted frame, we employed
the MTCNN (Multi-task Cascaded Convolutional Networks)
to detect and crop the faces. MTCNN is a popular face
detection algorithm that efficiently localizes facial regions
while handling variations in pose and lighting. Each
detected face was resized to a uniform size of 224x224 pixels
to maintain consistency across the dataset. This preprocessing
pipeline helped in standardizing the input data, making it
suitable for training the deep learning model. The cropped
face images were saved for subsequent training.
During training, we applied data augmentation techniques
to increase the diversity and robustness of the model.
These augmentations simulate real-world variations, helping
the model generalize better. The augmentations included
random resized cropping, horizontal flipping, rotation, color
jittering, Gaussian blurring, affine transformations, grayscale
conversion, and normalization. These transformations mimic
different visual conditions, thereby reducing the risk of over-
fitting.

## Model Architecture and Methodology


<img width="1007" height="579" alt="deepfake_architecture_diagram_edited drawio (1)" src="https://github.com/user-attachments/assets/8a96c427-b39b-47d5-b824-705ea1b93e61" />

# Methodology

Deep learning models have shown remarkable performance
in the field of image analysis and classification. CNNs have
traditionally been the go-to architecture due to their ability
to capture localized spatial features effectively. How-
ever, when it comes to understanding complex patterns and
long-range dependencies in images, ViTs have emerged as
a more powerful alternative. In this study, we leverage
the strengths of both architectures by combining EfficientNet-
B0 for spatial feature extraction with a ViT for relational
modeling. This hybrid approach aims to efficiently capture
both fine-grained and global patterns inherent in deepfake
videos.
The core of our model architecture is a hybrid design
that integrates EfficientNet-B0 and a Vision Transformer
(ViT). EfficientNet-B0 acts as the feature extractor, providing
spatial representations, while the ViT component captures
long-range dependencies through self-attention.
The input to the model is a batch of face images. These
images are first processed through EfficientNet-B0, which
extracts spatial features in the form of 1,280 feature maps,
each of size 7x7, resulting in a tensor of shape [batch, 1280,
7, 7]. To make these features suitable for the transformer
encoder, we flatten the spatial dimensions, transforming the
tensor to [batch, 49, 1280], where 49 represents the number of
tokens derived from the flattened feature maps. This structure
can be understood by drawing a parallel to natural language
processing (NLP), where each token represents a word.
Here, each of the 49 tokens corresponds to a spatial patch of
the image.
Next, we reduce the dimensionality of each feature vector
from 1,280 to 384 using a linear layer. This results in a tensor
of shape [batch, 49, 384]. A learnable CLS (classification)
token is appended to the beginning of the sequence, forming
a tensor of shape [batch, 50, 384]. This token aggregates
information from the entire image and is ultimately used
for classification. Positional embeddings are then added to
retain spatial information since transformers inherently lack
the ability to capture positional relationships.
The resulting tensor is passed through six consecutive ViT
blocks. Each block comprises multi-head attention (MHA) and
feed-forward neural network layers. The MHA mechanism
allows the model to focus on various parts of the image
simultaneously, capturing complex interdependencies between
facial features. Residual connections are incorporated
within each ViT block to enhance gradient flow, preventing
the vanishing gradient problem. Dropout layers are added at
various stages to reduce overfitting and improve generalization.

The final representation from the CLS token is passed
through an MLP head comprising linear and GELU layers
to output the probability of the input being a deepfake. This
combination of EfficientNet for spatial encoding and ViT for
relational modeling leads to a robust architecture capable of
capturing both local and global patterns.

The loss function used for training is Binary Cross En
tropy with Logits (BCEWithLogitsLoss), which is well-
suited for binary classification tasks as it combines a sigmoid
layer with binary cross-entropy loss. The optimizer chosen
is AdamW, known for its effective weight decay, with
a learning rate of 1e-4 and weight decay of 1e-4. To further
optimize training, we use a learning rate scheduler that reduces
the learning rate by a factor of 0.5 when the validation loss
plateaus for three epochs, promoting efficient convergence.

## Inference Pipeline

<img width="750" height="434" alt="inferencepipeline drawio" src="https://github.com/user-attachments/assets/509fb69d-5cc5-43e5-8ded-d6aafc6eceb2" />

## Comparison of model performance with and without Transformer on the FaceForensics++ and Caleb DF datasets

<img width="1198" height="434" alt="Screenshot 2026-06-01 191550" src="https://github.com/user-attachments/assets/d29cd5a1-5956-4937-8dbc-84360f68cf58" />


## Performance of the proposed model at different batch sizes

<img width="1141" height="224" alt="Screenshot 2026-06-01 191607" src="https://github.com/user-attachments/assets/cecca43d-85a4-49c8-bb5f-420703f878d5" />




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




