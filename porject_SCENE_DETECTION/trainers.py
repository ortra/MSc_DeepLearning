import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, batch_size, optimizer_class, criterion,
                 learning_rate=0.001, momentum=0.9, weight_decay=0):
        self.model = model.to(device)
        self.batch_size = batch_size
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.criterion = criterion

    def train(self, train_dataset):

        self.model.train()

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        print_every = np.ceil(len(dataloader) / 10)

        trainable_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.optimizer_class == torch.optim.SGD:
            optimizer = self.optimizer_class(trainable_parameters, lr=self.learning_rate, momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer_class(trainable_parameters, lr=self.learning_rate,
                                             weight_decay=self.weight_decay)
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, categories) in enumerate(dataloader):
            images = images.to(device)
            categories = categories.to(device)
            optimizer.zero_grad()
            probs = self.model.forward(images)
            loss = self.criterion(probs, categories)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted_categories = torch.argmax(probs, 1)
            correct += (predicted_categories == categories.to(device)).sum().item()
            total += categories.size(0)

            if i % print_every == 0 or i == len(dataloader) - 1:
                avg_loss = running_loss / (i + 1)
                accuracy = 100 * correct / total
                print(f'Batch [{i + 1}/{len(dataloader)}] '
                      f'Avg Loss: {avg_loss:.4f} '
                      f'Acc: {accuracy:.2f}[%] ({correct}/{total})')

        return accuracy, avg_loss

