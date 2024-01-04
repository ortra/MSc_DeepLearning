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
This  script is used for plotting results
It loads a pre-trained model  The key functionalities of this script include:

Loading a pre-trained model: The script allows you to specify the model name, type, variant, and dataset class.

Loading a dataset: It loads a dataset for testing purposes, such as CamSDD or Places365, based on the specified dataset class and partition (e.g., training, validation, or test).

Loading a checkpoint: If a checkpoint path is provided, the script loads the model's weights from the checkpoint, enabling the evaluation of a specific model state.

Depending on the model variant (e.g., zero_shot, linear_probe, or LoRA), the script uses an appropriate tester class to plot the results  on the dataset.

"""


def parse_args():
    parser = argparse.ArgumentParser(description='plotting examples from the datasets')
    parser.add_argument('--model_name', '-m', default='blip_feature_extractor', type=str,
                        help='Model load: which model we want to load ')
    parser.add_argument('--model_type', '-t', default='base', type=str,
                        help='Model load: which type of model we want to load ')
    parser.add_argument('--variant', '-v', default='zero_shot', type=str,
                        help='Model variant: zero_shot or linear_probe or LoRA')
    parser.add_argument('--dataset_class', '-c', default='CamSDD', type=str,
                        help='Dataset class: CamSDD, Places365 , etc.')
    parser.add_argument('--dataset_part', '-d',
                        default='test', type=str,
                        help='Dataset: training, validation, or test.')
    parser.add_argument('--num_samples', '-n', default=100, type=int,
                        help='num of samples to plot')
    parser.add_argument('--checkpoint_path', default=None, type=str,
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
    model, vis_processors, txt_processors = load_model(args.model_name, args.model_type, args.variant, True,
                                                       num_categories)
    print("Model's device:", next(model.parameters()).device)
    dataset = load_dataset(args.dataset_class, args.dataset_part, vis_processors["eval"])

    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(device)))

    title = f"{args.dataset_class}{' ' + args.dataset_part if args.dataset_part else ''} dataset {args.model_name}" \
            f" {args.model_type} {args.variant}"

    dataloader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    # Prepare the inputs
    images, _ = next(iter(dataloader))
    images = images.to(device)
    with torch.no_grad():
        if args.variant == 'zero_shot':
            texts = [f"a photo of  {prompt}".replace("_", " ")
                     for prompt in dataset.prompts]
            text_embeds = model.process_text(texts)
            samples = {"image": images, "text_input": text_embeds}
            probs = model.forward(samples)
        else:
            probs = model.forward(images)

    values, indices = probs.topk(3)

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
                            y=[dataset.prompts[idx] for idx in indices[index].cpu().numpy()],
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
