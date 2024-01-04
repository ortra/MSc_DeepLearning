"""Utility methods and constants used throughout the project."""
import os
import torch
from torch import nn
import json
from dataclasses import dataclass
from common import OUTPUT_DIR
from scene_detection_datasets import CamSDD, Places365
from lavis.models import load_model_and_preprocess
from datetime import datetime
from models import (BlipZeroShot,
                    CLIPZeroShot,
                    ClipLinearProbe,
                    BlipLinearProbe,
                    Blip2ZeroShot,
                    Blip2LinearProbe,
                    ClipLoRA,
                    BlipLoRA,
                    Blip2LoRA)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataset_class_mapping = {
    'CamSDD': CamSDD,
    'Places365': Places365
    # Add more dataset classes and their corresponding classes as needed
}

model_class_mapping = {
    'clip_zero_shot': CLIPZeroShot,
    'clip_linear_probe': ClipLinearProbe,
    'clip_LoRA': ClipLoRA,
    'blip_feature_extractor_zero_shot': BlipZeroShot,
    'blip_feature_extractor_linear_probe': BlipLinearProbe,
    'blip_feature_extractor_LoRA': BlipLoRA,
    'blip2_feature_extractor_zero_shot': Blip2ZeroShot,
    'blip2_feature_extractor_linear_probe': Blip2LinearProbe,
    'blip2_feature_extractor_LoRA': Blip2LoRA,
    # Add more dataset classes and their corresponding classes as needed
}


# Define the load_dataset function to dynamically select the dataset class
def load_dataset(dataset_class: str, dataset_part: str = None, transform=None) -> torch.utils.data.Dataset:
    if dataset_class in dataset_class_mapping:
        dataset_class_instance = dataset_class_mapping[dataset_class]
        if dataset_part is None:
            print(f"loading dataset using {dataset_class} ")
            root_path = dataset_class  # Use the dataset class name as the root path if dataset_part is not provided
        else:
            print(f"loading dataset using {dataset_class}  {dataset_part}")
            root_path = os.path.join(dataset_class, dataset_part)
        dataset = dataset_class_instance(root_path=root_path, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset class: {dataset_class}")

    return dataset


def load_model(model_name: str, model_type: str, model_variant: str, is_eval, num_categories=None) -> nn.Module:
    """Load the model corresponding to the name given.

    Args:
        model_name: the name of the model
        model_type : the type of the model
        model_variant : the variant of the model - zeroshot or linear probe or LoRA
        is_eval : mode
        num_categories: for linear probe models
    Returns:
        model: the model initialized, and loaded to device.

    """
    base_model, vis_processors, txt_processors = load_model_and_preprocess(model_name,
                                                                           model_type,
                                                                           is_eval=is_eval,
                                                                           device=device)

    args = [base_model, num_categories] if num_categories is not None else [base_model]

    print(f"Building model {model_name} {model_type}")
    model = model_class_mapping[model_name + '_' + model_variant](*args)
    print(f"Number of trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Number of model Parameters: {sum(p.numel() for p in model.parameters())}")
    return model.to(device), vis_processors, txt_processors


def normalize(image):
    """Normalize an image pixel values to [0, ..., 1]."""
    return (image - image.min()) / (image.max() - image.min())


@dataclass
class LoggingParameters:
    """Data class holding parameters for logging."""
    model_name: str
    model_type: str
    model_variant: str
    learning_rate: str
    epochs: str
    batch_size: str


def write_output(logging_parameters: LoggingParameters, data: dict):
    """Write logs to json.

    Args:
        logging_parameters: LoggingParameters. Some parameters to log.
        data: dict. Holding a dictionary to dump to the output json.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    output_filename = f"{timestamp}_{logging_parameters.model_name}_" \
                      f"{logging_parameters.model_type}_" \
                      f"{logging_parameters.model_variant}.json"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)

    print(f"Writing output to {output_filepath}")
    # Load output file
    if os.path.exists(output_filepath):
        # pylint: disable=C0103
        with open(output_filepath, 'r', encoding='utf-8') as f:
            all_output_data = json.load(f)
    else:
        all_output_data = []

    # Add new data and write to file
    all_output_data.append(data)
    # pylint: disable=C0103
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_output_data, f, indent=4)
