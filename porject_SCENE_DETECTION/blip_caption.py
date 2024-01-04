import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from common import FIGURES_DIR
from utils import load_dataset, normalize
from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    model_name = "blip_caption"
    model_type = "base_coco"
    dataset_class = "CamSDD"
    dataset_type = "test"
    num_samples = 8

    # Assuming load_model_and_preprocess returns the model
    model, vis_processors, txt_processors = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)

    dataset = load_dataset(dataset_class, dataset_type, vis_processors["eval"])
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, categories = next(iter(dataloader))
    images = images.to(device)

    title = f"{dataset_class} {dataset_type} dataset {model_name} {model_type} "
    num_examples = num_samples
    num_cols = int(num_examples ** 0.5)
    num_rows = (num_examples + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=(5,5))

    with torch.no_grad():
        # Generate captions for the entire batch
        captions = (model.generate({"image": images}))
        captions = [txt_processors["eval"](caption) for caption in captions]

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < num_examples:
                axes[i, j * 2].imshow(normalize(images[index].cpu()).permute(1, 2, 0))
                axes[i, j * 2].axis("off")

                # Get the caption for the current index
                caption = captions[index]
                axes[i, j * 2 + 1].text(0.5, 0.5, caption, ha='center', va='center', wrap=True)
                axes[i, j * 2 + 1].axis("off")

    # Remove any empty subplots (if num_examples is not a perfect fit)
    for idx in range(num_examples * 2, num_rows * num_cols * 2):
        fig.delaxes(axes.flatten()[idx])
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(wspace=0.5)
    plt.show()

    fig.savefig(os.path.join(FIGURES_DIR, title))

if __name__ == "__main__":
    main()
