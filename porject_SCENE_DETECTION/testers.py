import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ZeroShotTester:
    def __init__(self, model, txt_processors, batch_size):
        self.model = model.to(device)
        self.txt_processors = txt_processors
        self.batch_size = batch_size

    def test(self, dataset):
        self.model.eval()
        correct_top1 = 0
        correct_top3 = 0
        total = 0

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

        print_every = np.ceil(len(dataloader) / 10)
        texts = [f"a photo of {prompt}".replace("_", " ")
                 for prompt in dataset.prompts]
        text_embeds = self.model.process_text(texts)

        with torch.no_grad():
            for i, (images, categories) in enumerate(dataloader):
                images = images.to(device)
                samples = {"image": images, "text_input": text_embeds}
                probs = self.model.forward(samples)
                top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
                predicted_top1 = top3_indices[:, 0]  # Top-1 prediction
                predicted_top3 = top3_indices  # Top-3 predictions

                correct_top1 += (predicted_top1 == categories.to(device)).sum().item()
                correct_top3 += (predicted_top3 == categories.to(device).unsqueeze(1)).sum().item()
                total += categories.size(0)
                accuracy_top1 = 100 * correct_top1 / total
                accuracy_top3 = 100 * correct_top3 / total
                if i % print_every == 0 or i == len(dataloader) - 1:
                    print(f'Batch [{i + 1}/{len(dataloader)}] '
                          f'Top-1 Acc: {accuracy_top1:.2f}[%] '
                          f'({correct_top1}/{total}) | '
                          f'Top-3 Acc: {accuracy_top3:.2f}[%] '
                          f'({correct_top3}/{total})')
        return accuracy_top1, accuracy_top3


class Tester:
    def __init__(self, model, batch_size):
        self.model = model.to(device)
        self.batch_size = batch_size

    def test(self, dataset):
        self.model.eval()
        correct_top1 = 0
        correct_top3 = 0
        total = 0

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4  # You can change this value or make it customizable
        )
        print_every = np.ceil(len(dataloader) / 10)
        with torch.no_grad():
            for i, (images, categories) in enumerate(dataloader):
                images = images.to(device)
                probs = self.model.forward(images)

                top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
                predicted_top1 = top3_indices[:, 0]  # Top-1 prediction
                predicted_top3 = top3_indices  # Top-3 predictions

                correct_top1 += (predicted_top1 == categories.to(device)).sum().item()
                correct_top3 += (predicted_top3 == categories.to(device).unsqueeze(1)).sum().item()
                total += categories.size(0)
                accuracy_top1 = 100 * correct_top1 / total
                accuracy_top3 = 100 * correct_top3 / total
                if i % print_every == 0 or i == len(dataloader) - 1:
                    print(f'Top-1 Acc: {accuracy_top1:.2f}[%] '
                          f'({correct_top1}/{total}) | '
                          f'Top-3 Acc: {accuracy_top3:.2f}[%] '
                          f'({correct_top3}/{total})')
        return accuracy_top1, accuracy_top3
