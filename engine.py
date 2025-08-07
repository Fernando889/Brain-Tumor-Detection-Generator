from timeit import default_timer as timer
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import accuracy_function, total_train_time, set_seed
from typing import Callable


def train_step(model: nn.Module,
               data_loader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               acc_fn: Callable = accuracy_function,
               device: torch.device = None):

    model.train()
    train_loss, train_acc = 0, 0

    for X, y in data_loader:

        X, y = X.to(device), y.to(device)

        train_preds = model(X)

        loss = loss_fn(train_preds, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_preds_labels = torch.softmax(train_preds, dim=1).argmax(dim=1)
        train_acc += acc_fn(train_preds_labels, y)

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def test_step(model: nn.Module,
              data_loader: DataLoader,
              loss_fn: nn.Module,
              acc_fn: Callable = accuracy_function,
              device: torch.device = None):

    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in data_loader:

            X, y = X.to(device), y.to(device)

            test_preds = model(X)

            loss = loss_fn(test_preds, y)
            test_loss += loss.item()

            test_preds_labels = torch.argmax(test_preds, dim=1)
            test_acc += acc_fn(test_preds_labels, y)

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    return test_loss, test_acc


def train(model: nn.Module,
          train_data_loader: DataLoader,
          test_data_loader: DataLoader,
          loss_fn: nn.Module,
          epochs: int,
          optimizer: torch.optim.Optimizer,
          acc_fn: Callable = accuracy_function,
          device: torch.device = None,
          print_time: Callable = total_train_time):

    set_seed()

    model.to(device)

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    start_time = timer()

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

        print(f"Number of Epochs: {epoch+1} |"
              f"Train Loss: {train_loss:.2f} |"
              f"Train Accuracy: {train_acc:.2f}% |"
              f"Test Loss: {test_loss:.2f} |"
              f"Test Accuracy: {test_acc:.2f}%")

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    end_time = timer()
    print_time(start_time, end_time, device)

    return results
