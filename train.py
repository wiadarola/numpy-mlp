from typing import Generator

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from tqdm import tqdm

from src import functional as F
from src.model import NumpyMLP


def collate(dataset: Dataset, batch_size: int) -> Generator[tuple[NDArray, NDArray]]:
    N = len(dataset)
    indices = np.random.choice(range(0, N), (N // batch_size, batch_size))

    for batch_indices in indices:
        imgs, targets = [], []
        for idx in batch_indices:
            img, target = dataset[idx]
            imgs.append(np.array(img))
            targets.append(target)
        yield np.stack(imgs), np.stack(targets)


@hydra.main(".", "config", None)
def main(cfg: DictConfig):
    writer = SummaryWriter(HydraConfig.get().run.dir)

    model = NumpyMLP(**cfg.model)

    train_set = MNIST("data/", download=True)
    valid_set = MNIST("data/", train=False, download=True)

    for epoch in tqdm(range(cfg.trainer.n_epochs)):
        total_loss = total_accuracy = 0
        for x, y in tqdm(
            collate(train_set, cfg.dataset.batch_size),
            "Training",
            leave=False,
            total=len(train_set),
        ):
            y_hat = model(x)
            model.backward(x, y, lr=cfg.trainer.lr)

            total_loss += F.categorical_cross_entropy(y, y_hat)
            total_accuracy += F.accuracy_score(y, y_hat)

        writer.add_scalar("loss/train", total_loss / len(train_set), epoch)
        writer.add_scalar("accuracy/train", total_accuracy / len(train_set), epoch)

        total_loss = total_accuracy = 0
        for x, y in tqdm(
            collate(valid_set, cfg.dataset.batch_size),
            "Validating",
            leave=False,
            total=len(valid_set),
        ):
            y_hat = model(x)

            total_loss += F.categorical_cross_entropy(y, y_hat)
            total_accuracy += F.accuracy_score(y, y_hat)

        writer.add_scalar("loss/valid", total_loss / len(valid_set), epoch)
        writer.add_scalar("accuracy/valid", total_accuracy / len(valid_set), epoch)

    writer.close()


if __name__ == "__main__":
    main()
