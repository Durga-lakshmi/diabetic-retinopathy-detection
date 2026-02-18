import hydra
from torch.utils.data import DataLoader
import torch
from datasets import get_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from datasets.idrid.dataset import IDRID


@hydra.main(version_base=None, config_path='config', config_name='default.yaml')
def compute_mean_std(cfg):

    full_dataset = IDRID(cfg, split='train')
    labels = torch.tensor([full_dataset[i][1] for i in range(len(full_dataset))]).numpy()

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_idx, val_idx = next(splitter.split(torch.zeros(len(labels)), labels))

    train_dataset, train_dataloader = get_dataset(cfg, train_idx, "train")

    mean = 0.
    std = 0.
    total = 0

    for imgs, _ in train_dataloader:
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std  += imgs.std(2).sum(0)
        total += batch_samples

    mean /= total
    std /= total

    # print BEFORE returning
    print("MEAN:", mean)
    print("STD:", std)

    return mean, std


if __name__ == "__main__":
    compute_mean_std()
