import argparse
import sys
from os.path import join

import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader


@click.group()
def cli():
    pass


def load(test_data: str, test_label: str) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.load(join("data", "processed", test_data)), torch.load(
        join("data", "processed", test_label)
    )


def accuracy(model, data, labels):
    out = model(data)
    predicted, actual = out.argmax(axis=1), labels.argmax(axis=1)
    correct = predicted == actual
    return float(sum(correct) / 5000)


@click.command()
@click.argument("model_checkpoint")
def predict(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = torch.load(model_checkpoint)
    test_data, test_label = load("test_data.tensor", "test_labels.tensor")
    print(f"Accuracy: {accuracy(model, test_data, test_label):%}")


cli.add_command(predict)

if __name__ == "__main__":
    cli()
