from torch.utils.tensorboard.writer import SummaryWriter
import os
from datetime import datetime
import torch
from pathlib import Path
from torch import nn


def accuracy_function(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/(len(y_pred)))*100
    return acc


def total_train_time(start: float,
                     end: float,
                     device: torch.device = None):

    if device is None:
        device = torch.device("cpu")

    total_time = end-start
    print(
        f"Total Time Train in {device}: {total_time/60:.2f} minutes or {total_time:.2f} seconds")
    return total_train_time


def create_writers(experiment_name: str,
                   model_name: str,
                   extra: str = None):

    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        log_dir = os.path.join(
            "runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
    print(f"Creating Summary Writer, Saving to {log_dir}")
    return SummaryWriter(log_dir=log_dir)


def saved_models(model: nn.Module,
                 class_names: list,
                 target_dir: str,
                 model_name: str):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_path = target_dir_path/model_name

    print(f"Model Save Path: {model_save_path}")
    print(f"Saving {model_name} to {model_save_path}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "model_name": "effnetb0"
    }, model_save_path)

    print("Model Save Succesfully!")


def set_seed(num_seed: int = 42):
    torch.cuda.manual_seed(num_seed)
    torch.manual_seed(num_seed)
