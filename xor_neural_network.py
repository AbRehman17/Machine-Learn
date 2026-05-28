import numpy as np
import matplotlib.pyplot as plt
# ================================================================
# XOR TRUTH TABLE
# ================================================================
x = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])
# for plotting
loss_history = []
# ================================================================
# KEEP TRYING UNTIL NETWORK LEARNS
# ================================================================
for attempt in range(10):
    
    # reinitialize weights every attempt
    layer1 = np.random.randn(2, 2)
    b1 = np.zeros((1, 2))
    layer2 = np.random.randn(2, 1)
    b2 = np.zeros((1, 1))

    learning_rate = 1
    epochs = 100000

    for i in range(epochs):

        # FORWARD PASS
        z1 = np.dot(x, layer1) + b1
        o1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(o1, layer2) + b2
        output = 1 / (1 + np.exp(-z2))

        # LOSS
        loss = (1/len(y)) * np.sum((output - y)**2)
        # For Matplot
        loss_history.append(loss)

        # BACKPROP
        error = output - y
        delta_out = error * (output * (1 - output))
        dlayer2 = np.dot(o1.T, delta_out)
        db2 = np.sum(delta_out, axis=0, keepdims=True)

        delta_hidden = np.dot(delta_out, layer2.T) * (o1 * (1 - o1))
        dlayer1 = np.dot(x.T, delta_hidden)
        db1 = np.sum(delta_hidden, axis=0, keepdims=True)

        # UPDATE WEIGHTS
        layer2 = layer2 - learning_rate * dlayer2
        layer1 = layer1 - learning_rate * dlayer1
        b2 = b2 - learning_rate * db2
        b1 = b1 - learning_rate * db1
    # check if network learned
    print(f"\nAttempt {attempt+1} | Final Loss: {loss:.6f}")
    
    if loss < 0.01:
        print("NETWORK LEARNED! ")
        break
    else:
        print("Stuck! Trying again with new weights...")

# FINAL PREDICTIONS
print("\nFinal Predictions vs True Values:")
for i in range(len(x)):
    print(f"Input: {x[i]} | Predicted: {output[i][0]:.4f} | True: {y[i][0]}")
plt.plot(loss_history)
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()