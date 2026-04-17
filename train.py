import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import LeNet5  # Importing the model we built

# 1. Device configuration
# Uses GPU if available, otherwise falls back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# 2. Hyperparameters
num_epochs = 20
batch_size = 64
learning_rate = 0.01

# 3. Data Preparation (Crucial for LeNet-5)
# We must pad the 28x28 MNIST images to 32x32 and convert them to tensors
transform = transforms.Compose([
    transforms.Pad(2), # Pads 2 pixels on all sides: 28 + 2 + 2 = 32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
])

# Download and load the training dataset
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transform,  
                                           download=True)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

# Download and load the test dataset for evaluation
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transform, 
                                          download=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# 4. Initialize Model, Loss Function, and Optimizer
model = (LeNet5(num_classes=10).to(device))

# CrossEntropyLoss includes the Softmax activation internally
criterion = nn.CrossEntropyLoss()

# Using SGD with momentum, closely aligning with the original paper's approach
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 5. Training Loop
total_steps = len(train_loader)
print("Starting training...")

epoch_losses = []


for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute new gradients
        optimizer.step()      # Update weights
        
        running_loss += loss.item()
        
        if (i + 1) % 400 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")

    epoch_losses.append(loss)
    # Evaluate the model on the test set after each epoch
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation for testing
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"End of Epoch {epoch+1} - Test Accuracy: {accuracy:.2f}%")

# 6. Save the Model Weights
# It is best practice to save the state_dict rather than the entire model object
torch.save(model.state_dict(), 'lenet5_mnist_weights.pth')
print("Training complete! Model weights saved to 'lenet5_mnist_weights.pth'.")