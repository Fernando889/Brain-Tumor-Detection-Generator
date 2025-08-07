from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

NUM_WORKERS = os.cpu_count()


def create_data_loader(train_dir: str,
                       test_dir: str,
                       transforms: transforms,
                       batch_size: int = 32):

    train_data = ImageFolder(root=train_dir, transform=transforms)
    test_data = ImageFolder(root=test_dir, transform=transforms)

    class_names = train_data.classes
    print(f"Class Names: {class_names}")

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=NUM_WORKERS,
                                   pin_memory=True)

    test_data_loader = DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True)

    return train_data_loader, test_data_loader, class_names
