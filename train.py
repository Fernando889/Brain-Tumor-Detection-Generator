import torch
from torch import nn
from torchvision import transforms
from data_setup import create_data_loader
from engine import train
from rich import print
from torchinfo import summary
from models import create_effnetb0
from utils import saved_models
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = "brisc2025/classification_task/train"
test_dir = "brisc2025/classification_task/test"

manual_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data_loader, test_data_loaders, class_names = create_data_loader(train_dir=train_dir,
                                                                       test_dir=test_dir,
                                                                       transforms=manual_transforms)
model = create_effnetb0(out_features=len(class_names))
model.to(device)


model_summary = summary(model=model,
                        input_size=[1, 3, 224, 224],
                        col_names=["input_size", "output_size",
                                   "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"])

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

model_result = train(model=model,
                     train_data_loader=train_data_loader,
                     test_data_loader=test_data_loaders,
                     loss_fn=loss_fn,
                     epochs=40,
                     optimizer=optimizer,
                     device=device)

print(f"Model Result: {model_result}")

saved_models(model=model,
             class_names=class_names,
             target_dir="saved_models",
             model_name="effnetb0.pth")
