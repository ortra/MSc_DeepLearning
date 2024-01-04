import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from common import FIGURES_DIR
from utils import load_dataset, load_model, normalize
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
This script is used to visualize the results of combining the zero shot performance of different models (CLIP, BLIP, and BLIP2)
in classifying scene categories based on images and textual prompts,
selecting the prediction from the most confident model for each image."
"""


def parse_args():
    parser = argparse.ArgumentParser(description='plotting examples from the datasets')
    parser.add_argument('--dataset_class', '-c', default='Places365', type=str,
                        help='Dataset class: CamSDD, Places365 , etc.')
    parser.add_argument('--dataset_part', '-d',
                        default='validation', type=str,
                        help='Dataset: training, validation, or test.')
    parser.add_argument('--num_samples', '-n', default=100, type=int,
                        help='num of samples to plot')
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

    title = "Mixed"
    torch.manual_seed(42)
    dataloader_clip = DataLoader(dataset_clip, batch_size=args.num_samples, shuffle=True, generator=torch.Generator().manual_seed(42))
    dataloader_blip = DataLoader(dataset_blip, batch_size=args.num_samples, shuffle=True, generator=torch.Generator().manual_seed(42))
    dataloader_blip2 = DataLoader(dataset_blip2, batch_size=args.num_samples, shuffle=True, generator=torch.Generator().manual_seed(42))

    texts = [f"a photo of  {prompt}".replace("_", " ")
             for prompt in dataset_clip.prompts]

    images, _ = next(iter(dataloader_clip))
    images = images.to(device)
    with torch.no_grad():
        text_embeds = clip_model.process_text(texts)
        samples = {"image": images, "text_input": text_embeds}
        clip_probs = clip_model.forward(samples)
    images, _ = next(iter(dataloader_blip))
    images = images.to(device)
    with torch.no_grad():
        text_embeds = blip_model.process_text(texts)
        samples = {"image": images, "text_input": text_embeds}
        blip_probs = blip_model.forward(samples)
    images, _ = next(iter(dataloader_blip2))
    images = images.to(device)
    with torch.no_grad():
        text_embeds = blip2_model.process_text(texts)
        samples = {"image": images, "text_input": text_embeds}
        blip2_probs = blip2_model.forward(samples)

    similarity = torch.cat((clip_probs.unsqueeze(1), blip_probs.unsqueeze(1), blip2_probs.unsqueeze(1)), dim=1)
    print(similarity.shape)
    similarity = filter_similarity(similarity)
    print(similarity.shape)
    values, indices = similarity.topk(3)

    num_examples = args.num_samples
    num_cols = int(num_examples ** 0.5)
    num_rows = (num_examples + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=(5, 5))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j  # Convert 2D index (i, j) to 1D index
            if index < num_examples:  # Assuming you have defined num_examples
                axes[i, j * 2].imshow(normalize(images[index].cpu()).permute(1, 2, 0))
                axes[i, j * 2].axis("off")

                # Plot the bar chart using Seaborn
                sns.barplot(x=values[index].cpu().numpy(),
                            y=[dataset_clip.prompts[idx] for idx in indices[index].cpu().numpy()],
                            ax=axes[i, j * 2 + 1], width=0.3)

                # Adding percentage annotations to each bar with a smaller font size
                for bar, percentage in zip(axes[i, j * 2 + 1].patches, values[index].cpu()):
                    width = bar.get_width()
                    axes[i, j * 2 + 1].annotate(f'{percentage * 100:.3f}%', (width, bar.get_y() + bar.get_height() / 2),
                                                va='center', fontsize=8, color='black')

                # Remove the axes of the graph
                sns.despine(bottom=True, left=True, ax=axes[i, j * 2 + 1])
                axes[i, j * 2 + 1].xaxis.set_visible(False)
                # Decrease font size for category names and adjust the width of the graph
                axes[i, j * 2 + 1].tick_params(axis='y', labelsize=8)

    # Remove any empty subplots (if num_examples is not a perfect fit)
    for idx in range(num_examples * 2, num_rows * num_cols * 2):
        fig.delaxes(axes.flatten()[idx])
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(wspace=0.5)
    plt.show()

    fig.savefig(os.path.join(FIGURES_DIR, title))


if __name__ == "__main__":
    main()
