# -*- coding: utf-8 -*-
import logging
from os import listdir
from os.path import join
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


def onehot(v: int):
    a = [0] * 10
    a[v] = 1
    return a


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train: torch.Tensor = None
    train_label: torch.Tensor = None
    for file in listdir(input_filepath):
        if file == ".gitkeep":
            continue
        data = np.load(join(input_filepath, file), allow_pickle=True)
        if file.startswith("train"):
            if train is None:
                train = torch.as_tensor(
                    data["images"].reshape((-1, 28 * 28)), dtype=float
                )
                train_label = torch.as_tensor(
                    [onehot(v) for v in data["labels"]], dtype=float
                )
            else:
                train = torch.cat(
                    (
                        train,
                        torch.as_tensor(
                            data["images"].reshape((-1, 28 * 28)), dtype=float
                        ),
                    )
                )
                train_label = torch.cat(
                    (
                        train_label,
                        torch.as_tensor(
                            [onehot(v) for v in data["labels"]], dtype=float
                        ),
                    )
                )
        else:
            test: torch.Tensor = torch.as_tensor(
                data["images"].reshape((-1, 28 * 28)), dtype=float
            )
            test_label: torch.Tensor = torch.as_tensor(
                [onehot(v) for v in data["labels"]], dtype=float
            )

    train_std = (train - train.mean()) / train.std()
    test_std = (test - test.mean()) / test.std()
    torch.save(train_std, join(output_filepath, "train_data.tensor"))
    torch.save(test_std, join(output_filepath, "test_data.tensor"))
    torch.save(train_label, join(output_filepath, "train_labels.tensor"))
    torch.save(test_label, join(output_filepath, "test_labels.tensor"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
