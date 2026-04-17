import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Layer C1: Convolutional (6 filters, 5x5)
        # Input: 1 channel (Grayscale), Output: 6 channels
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        
        # Layer S2: Max Pooling (2x2, stride 2)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer C3: Convolutional (16 filters, 5x5)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        # Layer S4: Max Pooling (2x2, stride 2)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer C5: Flatten then Fully Connected (or 5x5 Conv)
        # After S4, the feature map is 5x5. 16 * 5 * 5 = 400
        self.c5 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        
        # Layer F6: Fully Connected
        self.f6 = nn.Linear(in_features=120, out_features=84)
        
        # Output Layer
        self.output = nn.Linear(in_features=84, out_features=num_classes)
        
        # Modern Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional blocks
        x = self.relu(self.c1(x))
        x = self.s2(x)
        
        x = self.relu(self.c3(x))
        x = self.s4(x)
        
        # Flattening for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected blocks
        x = self.relu(self.c5(x))
        x = self.relu(self.f6(x))
        
        # Final output (Softmax is usually handled by the Loss Function in PyTorch)
        x = self.output(x)
        
        return x

# Instantiate the model
model = LeNet5(num_classes=10)
print(model)
