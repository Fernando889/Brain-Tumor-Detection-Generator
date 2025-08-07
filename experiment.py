import torch
from torch import nn
from torchvision import transforms
from data_setup import create_data_loader
from typing import Dict, List, Callable
from torch.utils.data import DataLoader
from utils import total_train_time, set_seed, create_writers, accuracy_function
from timeit import default_timer as timer
from models import create_effnetb0, create_effnetb2
from engine_writer import train_writer


def experiment(models: List[str],
               num_epochs: List[int],
               train_data_loader: DataLoader,
               test_data_loader: DataLoader,
               out_features: int,
               device: torch.device = None,
               print_time: Callable = total_train_time):

    set_seed()
    time_start = timer()

    experiment_number = 0
    data_loader_name = "brisc2025"

    for epochs in num_epochs:
        for models_name in models:
            experiment_number += 1
            print(f"Experiment Number: {experiment_number}")
            print(f"Model Name: {models_name}")
            print(f"DataLoaders Name: {data_loader_name}")
            print(f"Number of Epochs: {epochs}")

            if models_name == "effnetb0":
                model = create_effnetb0(out_features=out_features)
            else:
                model = create_effnetb2(out_features=out_features)

            model.to(device)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            train_writer(model=model,
                         train_data_loader=train_data_loader,
                         test_data_loader=test_data_loader,
                         loss_fn=loss_fn,
                         optimizer=optimizer,
                         epochs=epochs,
                         writer=create_writers(experiment_name=data_loader_name,
                                               model_name=models_name,
                                               extra=f"{epochs}_epochs"),
                         acc_fn=accuracy_function,
                         device=device,
                         print_time=total_train_time)

    time_end = timer()
    print_time(time_start, time_end, device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device Use: {device}")

train_dir = "brisc2025/classification_task/train"
test_dir = "brisc2025/classification_task/test"

manual_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data_loaders, test_data_loaders, class_names = create_data_loader(train_dir=train_dir,
                                                                        test_dir=test_dir,
                                                                        transforms=manual_transforms)


experiment(models=["effnetb0", "effnetb2"],
           num_epochs=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
           train_data_loader=train_data_loaders,
           test_data_loader=test_data_loaders,
           out_features=len(class_names),
           device=device)
