"""Main training script."""
import argparse
import torch
import os
from common import CHECKPOINT_DIR
from testers import Tester
from trainers import Trainer
from utils import load_dataset, load_model, LoggingParameters, write_output
from datetime import datetime
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

"""
This  script is used for testing the  models
It loads a pre-trained model, and evaluates its performance on a given dataset. The key functionalities of this script include:

Loading a pre-trained model: The script allows you to specify the model name, type, variant, and dataset class.

Loading a dataset: It loads a dataset for testing purposes, such as CamSDD or Places365, based on the specified dataset class and partition (e.g., training, validation, or test).

Loading a checkpoint: If a checkpoint path is provided, the script loads the model's weights from the checkpoint, enabling the evaluation of a specific model state.

Training the model: Depending on the model variant (e.g. linear_probe, or LoRA), the script uses an appropriate tester class to train the model on a given dataset.
 This involves running inference on the dataset and reporting the evaluation results.
"""

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='Testing and training a model')
    parser.add_argument('--model_name', '-m', default='blip_feature_extractor', type=str,
                        help='Model name to load: clip, blip_feature_extractor, blip2_feature_extractor')
    parser.add_argument('--model_type', '-t', default='base', type=str,
                        help='Model type to load: ViT-B-32, base, pretrain')
    parser.add_argument('--variant', '-v', default='LoRA', type=str,
                        help='Model variant: linear_probe')
    parser.add_argument('--dataset_class', '-c', default='CamSDD', type=str,
                        help='Dataset class: CamSDD, Places365 etc.')
    parser.add_argument('--batch_size', '-b', default=128, type=int,
                        help='Batch size for testing')
    parser.add_argument('--learning_rate', '-lr', default=0.001, type=float,
                        help='The learning rate')
    parser.add_argument('--optimizer', '-opt', default='Adam', choices=['Adam', 'AdamW', 'SGD'], type=str,
                        help='Model optimizer')
    parser.add_argument('--momentum', '-mom', default=0.9, type=float,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', '-wd', default=0.0, type=float,
                        help='Weight decay for the optimizer')
    parser.add_argument('--epochs', default=3, type=int,
                        help='Number of epochs to train')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Path to checkpoint for model loading')

    return parser.parse_args()


def main():
    """Parse arguments and train model on dataset."""
    args = parse_args()
    category_mapping = {
        'zero_shot': {'CamSDD': None, 'Places365': None},
        'linear_probe': {'CamSDD': 30, 'Places365': 365},
        'LoRA': {'CamSDD': 30, 'Places365': 365}
    }
    num_categories = category_mapping[args.variant][args.dataset_class]
    model, vis_processors, txt_processors = load_model(args.model_name, args.model_type, args.variant, True,
                                                       num_categories)
    # Check the device of any parameter (they are all on the same device)
    print("Model's device:", next(model.parameters()).device)

    train_dataset = load_dataset(args.dataset_class, 'training', vis_processors["eval"])
    validation_dataset = load_dataset(args.dataset_class, 'validation', vis_processors["eval"])
    test_dataset = load_dataset(args.dataset_class, 'test', vis_processors["eval"])

    experiment_name = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{args.model_name}_{args.model_type}_{args.variant}"
    logdir = f'runs/{experiment_name}'

    writer = SummaryWriter(log_dir=logdir)

    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(device)))

    momentum = args.momentum
    weight_decay = args.weight_decay

    # Optimizer
    if args.optimizer == 'Adam':
        optimizer_class = torch.optim.Adam
        optimizer_params = {'weight_decay': weight_decay}
    elif args.optimizer == 'AdamW':
        optimizer_class = torch.optim.AdamW
        optimizer_params = {'weight_decay': weight_decay}
    elif args.optimizer == 'SGD':
        optimizer_class = torch.optim.SGD
        optimizer_params = {'momentum': momentum, 'weight_decay': weight_decay}
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    trainer = Trainer(model, batch_size, optimizer_class, criterion, learning_rate, **optimizer_params)

    tester = Tester(model, batch_size)
    logging_parameters = LoggingParameters(model_name=args.model_name,
                                           model_type=args.model_type,
                                           model_variant=args.variant,
                                           learning_rate=args.learning_rate,
                                           epochs=args.epochs,
                                           batch_size=args.batch_size)

    output_data = {
        "time_stamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "model_name": logging_parameters.model_name,
        "model_type": logging_parameters.model_type,
        "model_variant": logging_parameters.model_variant,
        "learning_rate": logging_parameters.learning_rate,
        "epochs": logging_parameters.epochs,
        "batch_size": logging_parameters.batch_size,
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }
    best_acc = 0

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    weights_filename = f"{timestamp}_{args.model_name}_" \
                       f"{args.model_type}_" \
                       f"{args.variant}_"\
                       f"{args.dataset_class}.pth"
    checkpoint_filename = os.path.join(CHECKPOINT_DIR, weights_filename)

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')

        print("training : ")
        train_acc, train_loss = trainer.train(train_dataset)
        print("validation : ")
        val_acc, _ = tester.test(validation_dataset)
        print("test : ")
        test_acc, _ = tester.test(test_dataset)
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        # JSON
        output_data["train_loss"].append(round(train_loss, 5))
        output_data["train_acc"].append(train_acc)
        output_data["val_acc"].append(val_acc)
        output_data["test_acc"].append(test_acc)

        if val_acc > best_acc:
            print(f'Saving checkpoint {checkpoint_filename}')
            torch.save(model.state_dict(), checkpoint_filename)
            best_acc = val_acc

    writer.close()

    # Create JSON
    write_output(logging_parameters, output_data)


if __name__ == '__main__':
    main()
