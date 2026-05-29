import torch
import torch.nn as nn

import torch

X = torch.tensor([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]], dtype=torch.float32)

y = torch.tensor([[0],
                  [1],
                  [1],
                  [0]], dtype=torch.float32)

print(X.shape)
print(y.shape)

#defining the model
class XOR_Network(nn.Module):
  def __init__(self) :
      super().__init__() # initializes pytorch internals 
      self.layer1=nn.Linear(2, 10)
      self.layer2=nn.Linear(10,1)
      self.sigmoid=nn.Sigmoid()
  def forward(self,x):
      x=self.layer1(x)
      x=self.sigmoid(x)
      x=self.layer2(x)
      x=self.sigmoid(x)
      return x
model=XOR_Network()
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
best_loss = float('inf')
patience = 500
counter = 0
min_improvement = 0.00001

for epoch in range(50000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if best_loss - loss.item() > min_improvement:
        best_loss = loss.item()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

print("\nFinal Predictions:")
print(model(X).detach())