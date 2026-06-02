"""
MNIST Digit Classifier
Neural network built in PyTorch to recognize handwritten digits (0-9)
Architecture: 784 -> 256 -> 128 -> 64 -> 10
Test Accuracy: 97.71%
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ========== LOAD DATA ==========
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print("Training samples:", len(train_data))
print("Test samples:", len(test_data))

# Show one example
image, label = train_data[0]
print("Image shape:", image.shape)
print("Label:", label)

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.show()

# ========== CREATE BATCHES ==========
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print("Total batches:", len(train_loader))

# ========== BUILD NETWORK ==========
class MNISTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

model = MNISTNetwork()
print(model)

# ========== TRAIN ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    total_loss = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

# ========== TEST ACCURACY ==========
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")