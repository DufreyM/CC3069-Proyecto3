"""Arquitecturas CNN del Laboratorio 3, sin cambios."""
import torch
import torch.nn as nn


class CNNBasica(nn.Module):
    """Dos capas convolucionales con MaxPooling y dos capas densas."""

    def __init__(self, num_classes: int, img_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        with torch.no_grad():
            flat_size = self._features(torch.zeros(1, 3, img_size, img_size)).reshape(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.conv1(x)))
        return self.pool2(self.relu(self.conv2(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))


class CNNProfunda(nn.Module):
    """Tres bloques convolucionales con BatchNorm y Dropout."""

    def __init__(self, num_classes: int, img_size: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        with torch.no_grad():
            salida = self.block3(self.block2(self.block1(torch.zeros(1, 3, img_size, img_size))))
            flat_size = salida.reshape(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_size, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block3(self.block2(self.block1(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc2(self.dropout_fc(self.relu(self.fc1(x))))


MODELOS = {"basica": CNNBasica, "profunda": CNNProfunda}
