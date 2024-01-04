import torch
import os
import argparse
from utils import load_dataset, load_model
import tensorflow as tf
from testers import ZeroShotTester, Tester
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

"""
This  script is used for testing the  models
It loads a pre-trained model, and evaluates its performance on a given dataset. The key functionalities of this script include:

Loading a pre-trained model: The script allows you to specify the model name, type, variant, and dataset class.

Loading a dataset: It loads a dataset for testing purposes, such as CamSDD or Places365, based on the specified dataset class and partition (e.g., training, validation, or test).

Loading a checkpoint: If a checkpoint path is provided, the script loads the model's weights from the checkpoint, enabling the evaluation of a specific model state.

Testing the model: Depending on the model variant (e.g., zero_shot, linear_probe, or LoRA), the script uses an appropriate tester class to evaluate the model's performance on the dataset.
 This involves running inference on the dataset and reporting the evaluation results.
"""

"""
the available checkpoints combinations are :

clip_ViT-B-32_zero_shot_CamSDD
blip_feature_extractor_base_zero_shot_CamSDD 
blip2_feature_extractor_pretrain_zero_shot_CamSDD
clip ViT-B-32 linear_probe_CamSDD  
blip_feature_extractor_base_linear_probe_CamSDD 
blip2_feature_extractor_pretrain_linear_probe_CamSDD 
clip ViT-B-32_LoRA_CamSDD
blip_feature_extractor_base_LoRA_CamSDD
blip2_feature_extractor_base_LoRA_CamSDD 

clip_ViT-B-32_zero_shot_Places365
blip_feature_extractor_base_zero_shot_Places365
blip2_feature_extractor_pretrain_zero_shot_Places365  
clip ViT-B-32 linear_probe_Places365  
clip ViT-B-32_LoRA_Places365
"""


def parse_args():
    parser = argparse.ArgumentParser(description='testing model')
    parser.add_argument('--model_name', '-m', default='blip2_feature_extractor', type=str,
                        help='Model load: which model we want to load ')
    parser.add_argument('--model_type', '-t', default='pretrain', type=str,
                        help='Model load: which type of model we want to load ')
    parser.add_argument('--variant', '-v', default='linear_probe', type=str,
                        help='Model variant: zero_shot or linear_probe or LoRA')
    parser.add_argument('--dataset_class', '-c', default='CamSDD', type=str,
                        help='Dataset class: CamSDD, Places365, etc.')
    parser.add_argument('--dataset_part', '-d',
                        default='test', type=str,
                        help='Dataset: training, validation, or test.')
    parser.add_argument('--batch_size', '-b', default=128, type=int,
                        help='Batch size for testing.')
    parser.add_argument('--checkpoint_path', default='checkpoints/20230909160127_blip2_feature_extractor_pretrain_linear_probe_CamSDD.pth', type=str,
                        help='Path to checkpoint for model loading')
    return parser.parse_args()


def main():

    args = parse_args()
    category_mapping = {
        'zero_shot': {'CamSDD': None, 'Places365': None},
        'linear_probe': {'CamSDD': 30, 'Places365': 365},
        'LoRA': {'CamSDD': 30, 'Places365': 365}
    }
    num_categories = category_mapping[args.variant][args.dataset_class]
    model, vis_processors, txt_processors = load_model(args.model_name, args.model_type, args.variant, True, num_categories)
    # Check the device of any parameter (they are all on the same device)
    print("Model's device:", next(model.parameters()).device)
    dataset = load_dataset(args.dataset_class, args.dataset_part, vis_processors["eval"])

    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(device)))

    tester_mapping = {
        'zero_shot': {
            'class': ZeroShotTester,
            'args': (txt_processors, args.batch_size)
        },
        'linear_probe': {
            'class': Tester,
            'args': (args.batch_size,)
        },
        'LoRA': {
            'class': Tester,
            'args': (args.batch_size,)
        }
    }

    tester_args = tester_mapping[args.variant]['args']
    tester = tester_mapping[args.variant]['class'](model, *tester_args)
    tester.test(dataset)


if __name__ == "__main__":
    main()



