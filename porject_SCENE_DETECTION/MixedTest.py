import torch
import os
import argparse
from utils import load_dataset, load_model
import tensorflow as tf
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

"""
This script is used to test the results of combining the zero shot performance of different models (CLIP, BLIP, and BLIP2)
in classifying scene categories based on images and textual prompts,
selecting the prediction from the most confident model for each image."
"""


def parse_args():
    parser = argparse.ArgumentParser(description='testing model')
    parser.add_argument('--dataset_class', '-c', default='CamSDD', type=str,
                        help='Dataset class: CamSDD, Places365, etc.')
    parser.add_argument('--dataset_part', '-d',
                        default='test', type=str,
                        help='Dataset: training, validation, or test.')
    parser.add_argument('--batch_size', '-b', default=128, type=int,
                        help='Batch size for testing.')
    return parser.parse_args()


def filter_similarity(similarity):
    max_values, _ = torch.max(similarity.view(similarity.size(0), -1), dim=1)
    max_values = max_values.view(-1, 1, 1)
    max_mask = (similarity == max_values)
    indices = torch.nonzero(max_mask, as_tuple=False)[:, :2]
    indices = filter_tensor_by_unique_first_value(indices)
    similarity = similarity[indices[:, 0], indices[:, 1]]
    return similarity


def filter_tensor_by_unique_first_value(original_tensor):
    original_list = original_tensor.tolist()
    unique_first_values = set()
    filtered_list = []
    for row in original_list:
        first_value = row[0]
        if first_value not in unique_first_values:
            filtered_list.append(row)
            unique_first_values.add(first_value)
    filtered_tensor = torch.tensor(filtered_list)
    return filtered_tensor


def main():

    args = parse_args()
    clip_model, clip_vis_processors, clip_txt_processors = load_model('clip', 'ViT-B-32', 'zero_shot', True)
    blip_model, blip_vis_processors, blip_txt_processors = load_model('blip_feature_extractor', 'base', 'zero_shot', True)
    blip2_model, blip2_vis_processors, blip2_txt_processors = load_model('blip2_feature_extractor', 'pretrain', 'zero_shot', True)

    dataset_clip = load_dataset(args.dataset_class, args.dataset_part, clip_vis_processors["eval"])
    dataset_blip = load_dataset(args.dataset_class, args.dataset_part, blip_vis_processors["eval"])
    dataset_blip2 = load_dataset(args.dataset_class, args.dataset_part, blip2_vis_processors["eval"])

    dataloader_clip = torch.utils.data.DataLoader(dataset_clip, batch_size=args.batch_size, shuffle=False)
    dataloader_blip = torch.utils.data.DataLoader(dataset_blip, batch_size=args.batch_size, shuffle=False)
    dataloader_blip2 = torch.utils.data.DataLoader(dataset_blip2, batch_size=args.batch_size, shuffle=False)

    print_every = np.ceil(len(dataloader_clip) / 10)
    texts = [f"a photo of  {prompt}".replace("_", " ")
             for prompt in dataset_clip.prompts]
    texts_clip = clip_model.process_text(texts)
    texts_blip = blip_model.process_text(texts)
    texts_blip2 = blip2_model.process_text(texts)

    # print(texts)
    num_batches = len(dataloader_clip)
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for i, (clip_batch, blip_batch, blip2_batch) in enumerate(
                zip(dataloader_clip, dataloader_blip, dataloader_blip2)):
            # Extract images and categories
            clip_images, clip_categories = clip_batch
            clip_images = clip_images.to(device)
            blip_images, blip_categories = blip_batch
            blip_images = blip_images.to(device)
            blip2_images, blip2_categories = blip2_batch
            blip2_images = blip2_images.to(device)

            # Prepare data for models
            samples_clip = {"image": clip_images, "text_input": texts_clip}
            samples_blip = {"image": blip_images, "text_input": texts_blip}
            samples_blip2 = {"image": blip2_images, "text_input": texts_blip2}

            # Forward pass through models
            clip_probs = clip_model.forward(samples_clip)
            blip_probs = blip_model.forward(samples_blip)
            blip2_probs = blip2_model.forward(samples_blip2)

            # Calculate similarity
            similarity = torch.cat((clip_probs.unsqueeze(1), blip_probs.unsqueeze(1), blip2_probs.unsqueeze(1)), dim=1)

            # Apply filtering if necessary
            similarity = filter_similarity(similarity)

            # Calculate top-3 predictions
            top3_probs, top3_indices = torch.topk(similarity, 3, dim=1)
            predicted_top1 = top3_indices[:, 0]  # Top-1 prediction
            predicted_top3 = top3_indices  # Top-3 predictions

            # Update accuracy metrics
            correct_top1 += (predicted_top1 == clip_categories.to(device)).sum().item()
            correct_top3 += (predicted_top3 == clip_categories.to(device).unsqueeze(1)).sum().item()
            total += clip_categories.size(0)

            # Calculate and print accuracy
            accuracy_top1 = 100 * correct_top1 / total
            accuracy_top3 = 100 * correct_top3 / total
            if i % print_every == 0 or i == len(dataloader_clip) - 1:
                print(f'Batch [{i + 1}/{len(dataloader_clip)}] '
                      f'Top-1 Acc: {accuracy_top1:.2f}[%] '
                      f'({correct_top1}/{total}) | '
                      f'Top-3 Acc: {accuracy_top3:.2f}[%] '
                      f'({correct_top3}/{total})')

if __name__ == "__main__":
    main()



