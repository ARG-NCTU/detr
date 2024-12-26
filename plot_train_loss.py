import json
import matplotlib.pyplot as plt

# Initialize lists to store the train_loss values and epochs
train_loss_values = []
epochs = []

# Read the text file line by line
with open('output/boat-1115-600-epochs/log.txt', 'r') as file:
    for line in file:
        # Parse each line as JSON
        data = json.loads(line.strip())
        # Append the train_loss and epoch to respective lists
        train_loss_values.append(data["train_loss"])
        epochs.append(data["epoch"])

# Plot the train_loss curve
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_values, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Train Loss Curve')
plt.legend()
plt.grid()
plt.savefig('output/boat-1115-600-epochs/train_loss_curve.png')
