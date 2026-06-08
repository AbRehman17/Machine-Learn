# Machine Learning Projects

Building neural networks from scratch to understand fundamentals, then using PyTorch for real-world problems.

## Projects

### 1. XOR Neural Network (NumPy)

- Built from scratch with manual backpropagation and gradient descent
- No PyTorch, no TensorFlow — just NumPy and math

### 2. XOR Neural Network (PyTorch)

- Same problem, automated with PyTorch
- Added early stopping and learning rate tuning

### 3. MNIST Digit Classifier (Fully Connected)

- **97.63% accuracy** on 10,000 test images
- 4-layer network: 784 → 256 → 128 → 64 → 10
- Trained on 60,000 real handwritten digit images
- Batch training with DataLoader

### 4. MNIST Digit Classifier (CNN) 🚀

- **99.07% accuracy** — 2% improvement over fully connected network
- Convolutional Neural Network architecture:
  - Conv1: 1 → 32 channels (3x3 kernel, padding=1)
  - MaxPool: 2x2 (28x28 → 14x14)
  - Conv2: 32 → 64 channels (3x3 kernel, padding=1)
  - MaxPool: 2x2 (14x14 → 7x7)
  - Flatten: 3136 features
  - FC1: 3136 → 128
  - FC2: 128 → 10
- **Total parameters:** ~401,000 (6x fewer than FC network!)
- Learned to detect edges, curves, and patterns automatically
- Can predict real-world handwritten digits from photos

## Model Comparison

| Model           | Accuracy | Parameters | Type          |
| --------------- | -------- | ---------- | ------------- |
| Fully Connected | 97.63%   | ~2.5M      | Dense layers  |
| CNN             | 99.07%   | ~401K      | Convolutional |

**CNN cuts errors by 60%** (290 → 99 mistakes per 10,000 images)

## Key Takeaways

### Fully Connected Networks

- Treat images as flat lists of pixels
- Lose spatial relationships between neighboring pixels
- Many parameters, risk of overfitting

### Convolutional Neural Networks

- Preserve spatial structure of images
- Learn hierarchical features (edges → curves → shapes → digits)
- Fewer parameters, better generalization
- Industry standard for computer vision

## What I Learned

- Forward pass, backpropagation, gradient descent (implemented manually)
- PyTorch tensors, autograd, optimizers
- CrossEntropy loss for multi-class classification
- Real image data preprocessing and batch training
- Hyperparameter tuning (learning rate, network depth)
- **CNN concepts:** convolution, filters/kernels, pooling, feature maps
- How to preprocess real-world images for model prediction
- Model saving/loading and deployment

## How to Run

### Train the CNN model

```bash
python MNIST_CNN/MNIST_cnn.py --mode train
```
