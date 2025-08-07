from engine import train_step, test_step
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Callable
from utils import accuracy_function, total_train_time, set_seed
from torch.utils.tensorboard.writer import SummaryWriter
from timeit import default_timer as timer


def train_writer(model: nn.Module,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 loss_fn: nn.Module,
                 epochs: int,
                 optimizer: torch.optim.Optimizer,
                 writer=SummaryWriter,
                 acc_fn: Callable = accuracy_function,
                 device: torch.device = None,
                 print_time: Callable = total_train_time):

    set_seed()

    model.to(device)

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    writer.add_graph(model=model, input_to_model=torch.randn(
        32, 3, 224, 224).to(device))

    start_timer = timer()

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(model=model,
                                           data_loader=train_data_loader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           acc_fn=acc_fn,
                                           device=device)

        test_loss, test_acc = test_step(model=model,
                                        data_loader=test_data_loader,
                                        loss_fn=loss_fn,
                                        acc_fn=acc_fn,
                                        device=device)

        print(f"Epochs: {epoch+1} |"
              f"Train Loss: {train_loss:.2f} |"
              f"Train Acc: {train_acc:.2f}% |"
              f"Test Loss: {test_loss:.2f} |"
              f"Test Acc: {test_acc:.2f}% |")

        if writer:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)

            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc},
                               global_step=epoch)

            writer.close()
        else:
            pass

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    end_timer = timer()
    print_time(start_timer, end_timer, device)

    return results
