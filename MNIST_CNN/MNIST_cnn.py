"""
MNIST CNN - Train and Test
Achieves 99%+ accuracy on handwritten digits
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================
# MODEL ARCHITECTURE
# ============================================

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(self.relu(self.conv2(x)))  # 14 -> 7
        x = x.view(-1, 64 * 7 * 7)               # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================================
# LOAD DATA
# ============================================

print("Loading MNIST data...")
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ============================================
# CREATE MODEL
# ============================================

model = MNIST_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# TRAIN
# ============================================

print("\n" + "="*50)
print("TRAINING STARTED")
print("="*50)

for epoch in range(5):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/5, Loss: {avg_loss:.4f}")

# ============================================
# TEST
# ============================================

print("\n" + "="*50)
print("TESTING")
print("="*50)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
print("="*50)

# ============================================
# SAVE MODEL
# ============================================

torch.save(model.state_dict(), 'mnist_cnn.pth')
print("\n✅ Model saved as 'mnist_cnn.pth'")