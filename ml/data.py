import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def load_mnist_data(root='data', flatten=True, batch_size=32):
    if flatten:
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Lambda(lambda x: torch.flatten(x))]
        )
    else:
        transform = torchvision.transforms.ToTensor(),

    train_dataset = MNIST(root=root, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False,
                         download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader
