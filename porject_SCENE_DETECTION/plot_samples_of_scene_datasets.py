"""Plot samples from the dataset."""
import os
import argparse
import random
import matplotlib.pyplot as plt
from common import FIGURES_DIR
from utils import load_dataset

"""
script that plotting an examples from our dataset
"""


def parse_args():
    parser = argparse.ArgumentParser(description='plotting examples from the datasets')
    parser.add_argument('--num_examples', '-n', default=9, type=int,
                        help='number of examples to plot')
    parser.add_argument('--dataset_class', '-c', default='Places365', type=str,
                        help='Dataset class: CamSDD, Places365 , etc.')
    parser.add_argument('--dataset_part', '-d',
                        default='validation', type=str,
                        help='Dataset: training, validation, or test.')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_class, args.dataset_part)
    title = args.dataset_class + (args.dataset_part if args.dataset_part is not None else '')
    num_examples = args.num_examples
    num_rows = int(num_examples ** 0.5)
    num_cols = (num_examples + num_rows - 1) // num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5, 5))

    for i in range(num_rows):
        for j in range(num_cols):
            idx = random.randint(0, len(dataset) - 1)
            img, prompt_idx = dataset[idx]
            prompt = (dataset.prompts[prompt_idx] if prompt_idx != -1 else '')

            # The image is already in RGB format, so we can directly use it for plotting
            axes[i, j].imshow(img)
            axes[i, j].set_title(prompt)
            axes[i, j].axis('off')

    # Remove any empty subplots (if num_examples is not a perfect square)
    for idx in range(num_examples, num_rows * num_cols):
        fig.delaxes(axes.flatten()[idx])

    plt.subplots_adjust(top=0.85)  # Adjust the positioning of the title
    plt.suptitle(title, fontsize=16)  # Add the title to the figure
    plt.show()
    fig.savefig(os.path.join(FIGURES_DIR, title + ' samples.jpg'))


if __name__ == "__main__":
    main()
