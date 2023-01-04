import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from os.path import join
import matplotlib.pyplot as plt

@click.group()
def cli():
    pass

def accuracy(model, data, labels):
    out = model(data)
    predicted, actual = out.argmax(axis = 1), labels.argmax(axis = 1)
    correct = predicted == actual
    return float(sum(correct)/5000)


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=5, help='number of epochs used for training')
def train(lr, epochs):
    print("Training day and night")
    print(f"{lr = }")
    print(f"{epochs = }")

    model = MyAwesomeModel()
    train_data, train_label, test_data, test_label = mnist()
    print(train_data.shape, train_label.shape)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    train_loader = DataLoader(list(zip(train_data, train_label)), batch_size=100, shuffle=True)
    
    history = [0]*epochs
    val = [0]*epochs
    for epoch in range(epochs):
        for data, label in train_loader:
            out = model(data)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            history[epoch] += loss.detach().numpy()
        val[epoch] = accuracy(model, test_data, test_label)
        print(epoch, history[epoch], val[epoch])
    torch.save(model, "trained_model.pt")
    plt.plot(history)
    plt.show()
    


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = torch.load(model_checkpoint)
    _, __, test_data, test_label = mnist()
    print(f"Accuracy: {accuracy(model, test_data, test_label):%}")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    