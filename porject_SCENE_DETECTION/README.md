
# SCENE DETECTION ALGORITHM EVALUATION BY VISION-LANGUAGE MODELS

## Description
This library represents the culmination of our project to train datasets for scene detection tasks. Using this library, users can train on the CamSDD Dataset with LoRA, perform zero-shot scene detection, generate various commands, and connect to Google Drive for project management.

## Abstract of the Study
This project explores the domain of Scene Detection, a crucial technique in computational photography that autonomously categorizes scenes within images, such as landscapes, portraits, and various lighting conditions. Our approach evaluates three pre-trained vision-language models—CLIP, BLIP, and BLIP-2—leveraging TOP-1 and TOP-3 accuracy metrics on the Camera Scene Detection Dataset (CamSDD), which comprises over 11,000 images in 30 categories, and the extensive Places365 dataset.

The models were tested as zero-shot classifiers and fine-tuned using linear probe and LoRA techniques. Our findings reveal that the Linear Probe approach significantly enhances the models' performance, with CLIP, BLIP, and BLIP-2 achieving TOP-1 accuracies up to 97.17% and a perfect TOP-3 accuracy of 100.00%. These results outperform traditional reference models like EfficientNet-B0, MobileNet-V2, and Xception, showcasing the advanced capability of vision-language models in improving image quality.

The study also highlights the potential of zero-shot configurations to provide a baseline for scene detection capabilities prior to any fine-tuning. Overall, our work underscores the effectiveness of Linear Probe and LoRA techniques in advancing scene detection algorithms, setting a precedent for future enhancements in photographic technology.

![Results](https://github.com/naorJR/SceneDetection/blob/master/figures/Results.jpg)

## Installation Instructions
To get started, install the necessary packages using the following commands:
```bash
!pip install salesforce-lavis
!pip install transformers
!pip install diffusers==0.18.1
!pip install tensorboardX
!pip install -U typing_extensions install transformers accelerate evaluate datasets peft -q
!pip install shutil
import torch
from PIL import Image
import diffusers
from lavis.models import load_model_and_preprocess

# To train the blip with lora, also run:
!pip install transformers==4.25
```

## (Optional) Support for Places365 Dataset

This library also offers support for the Places365 dataset from Kaggle, providing an additional resource for training scene detection models. To use this dataset, follow these instructions:

```bash
!pip install kaggle
```

**INTENTION** 
BEFORE RUNNING THE FOLLOWING CODE, YOU MUST PUT YOUR KAGGLE KEY IN THE GOOGLE DRIVE's PROJECT FOLDER!

```bash
import shutil
import os

# Create the .kaggle directory in /root/ if it doesn't exist
os.makedirs("/root/.kaggle", exist_ok=True)

# Copy the Kaggle API key from Google Drive to Colab
shutil.copy("/content/drive/My Drive/DL/project/SceneDetection-master/kaggle.json", "/root/.kaggle/kaggle.json")

# Set appropriate permissions
!chmod 600 /root/.kaggle/kaggle.json
```

```bash
# Download and unzip Places365
!kaggle datasets download -d benjaminkz/places365 -p "/content/"
!unzip /content/places365.zip -d /content/Places365/
!rm places365.zip
```

Ensure you have the necessary permissions and access to the Kaggle dataset for successful integration.

## Usage
Here are some examples of how you can use the library:

### Training Examples
#### Fine Tuning: Linear Probe, Dataset: CamSDD
- **Blip-2 Model Training Using Linear Probe with CamSDD**
  ```bash
  !python train.py --model_name blip2_feature_extractor --model_type pretrain --variant linear_probe --batch_size 32 --learning_rate 0.001 --optimizer Adam --momentum 0.9 --weight_decay 0.0 --epochs 5 #--checkpoint_path /path/to/your/checkpoint.pth
  ```

- **Blip Model Training Using Linear Probe with CamSDD**
  ```bash
  !python train.py --model_name blip_feature_extractor --model_type base --variant linear_probe --batch_size 32 --learning_rate 0.001 --optimizer Adam --momentum 0.9 --weight_decay 0.0 --epochs 5 #--checkpoint_path /path/to/your/checkpoint.pth
  ```

- **Clip Model Training Using Linear Probe with CamSDD**
  ```bash
  !python train.py --model_name clip --model_type ViT-B-32 --variant linear_probe --batch_size 32 --learning_rate 0.001 --optimizer Adam --momentum 0.9 --weight_decay 0.0 --epochs 5 #--checkpoint_path /path/to/your/checkpoint.pth
  ```

#### Fine Tuning: LoRA, Dataset: Places365
- **Blip-2 Model Training Using LoRA with CamSDD**
  ```bash
  !python train.py --model_name blip2_feature_extractor --model_type pretrain --variant LoRA --dataset_class CamSDD --batch_size 16 --learning_rate 0.001 --optimizer Adam --epochs 5
  ```

- **Blip Model Training Using LoRA with CamSDD**
  ```bash
  !pip install transformers==4.25
  ```
  ```bash
  !python train.py --model_name blip_feature_extractor --model_type base --variant LoRA --dataset_class CamSDD --batch_size 16 --learning_rate 0.001 --optimizer Adam --epochs 5
  ```

- **Clip Model Training Using LoRA with CamSDD**
  ```bash
  !python train.py --model_name clip --model_type ViT-B-32 --variant LoRA --dataset_class CamSDD --batch_size 128 --learning_rate 0.001 --optimizer Adam --epochs 5
  ```

#### Fine Tuning: LoRA, Dataset: Places365
- **Blip Model Training Using LoRA with Places365**
  ```bash
  !pip install transformers==4.25
  ```
  ```bash
  !python train.py --model_name blip2_feature_extractor --model_type pretrain --variant LoRA --dataset_class Places365 --batch_size 128 --learning_rate 0.001 --optimizer Adam --epochs 10
  ```

- **Clip Model Training Using LoRA with Places365**
  ```bash
  !python train.py --model_name clip --model_type ViT-B-32 --variant LoRA --dataset_class Places365 --batch_size 128 --learning_rate 0.001 --optimizer Adam --epochs 2
  ```

### Zero-shot Scene Detection Examples
- **Clip**
  ```bash
  !python test.py --model_name clip --model_type ViT-B-32 --variant zero_shot --dataset_class CamSDD --dataset_part validation --batch_size 128 
  ```

- **Blip**
  ```bash
  !python test.py --model_name blip_feature_extractor --model_type base --variant zero_shot --dataset_class validation --dataset_class CamSDD --batch_size 128
  ```

- **Blip-2**
  ```bash
  !python test.py --model_name blip2_feature_extractor --model_type pretrain --variant zero_shot --dataset_class validation --dataset_class CamSDD --batch_size 128 
  ```

### Testing Model with Pretrained Weights Examples
Notice: Here, it is the same commands as in the Zero-shot section above, but with a checkpoint_path field
- **Clip**
  ```bash
  !python test.py --model_name clip --model_type ViT-B-32 --variant zero_shot --dataset_class CamSDD --dataset_part validation --batch_size 128 --checkpoint_path /path/to/your/checkpoint.pth
  ```

### Utility Commands
- Lavis Models List
  ```bash
  !python available_models.py
  ```
- Plot Results
  ```bash
  !pip install transformers==4.25
  ```
  ```bash
  !python PlotResults.py
  ```
  ```bash
  !python plot_results_summary.py
  ```
  - BLIP Caption
  ```bash
  !python blip_caption.py
  ```
  - Launch Tensorboard
  ```bash
  !python launch_tensorboard.py
  ```

### Connecting to Google Drive
```python
from google.colab import drive
drive.mount('/content/drive/')
%cd '/content/drive/My Drive/DL/project/SceneDetection-master/'
import os
path = os.getcwd()
print('path: ' + path)
```

## Features
- Multiple Model Training: Supports training with various advanced vision-language models such as Blip-2, Blip, and Clip, offering flexibility in choosing the right model for specific scene detection tasks.

- LoRA Integration: Leveraging LoRA (Low-Rank Adaptation) for efficient and effective training, enhancing model adaptability and performance on the CamSDD Dataset.

- Zero-Shot Scene Detection: Provides the capability for zero-shot learning, allowing the models to recognize scenes without explicit example-based training, enhancing the versatility of the library.

## License
No License

## Contact Information
For any inquiries, feel free to reach out to us:
- Ort Trabelsi: ortrabelsi@mail.tau.ac.il
- Naor Cohen: naorcohen1@mail.tau.ac.il


