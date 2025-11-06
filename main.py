import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# Create folders if missing
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 1. Data Loading and Exploration
# -----------------------------
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transforms)
test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

print("Training data shape:", train_data.data.shape)
print("Test data shape:", test_data.data.shape)

# Show one test image
image, label = test_data[0]
plt.imshow(image.permute(1, 2, 0))
plt.title(f"Label: {label}")
plt.show()

# -----------------------------
# 2. Model Design
# -----------------------------
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)

# -----------------------------
# 3. Training Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImageClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
train_losses = []

# -----------------------------
# 4. Training Loop
# -----------------------------
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(trainloader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Plot and save loss curve
plt.plot(range(epochs), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss per Epoch')
plt.savefig('outputs/loss_plot.png')
plt.close()

# -----------------------------
# 5. Model Evaluation
# -----------------------------
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# -----------------------------
# 6. Save Model
# -----------------------------
torch.save(model.state_dict(), 'models/model.pth')
print("Model saved to models/model.pth")

# -----------------------------
# 7. Decision / Report
# -----------------------------
if accuracy >= 45:
    print("Model meets the performance target. Recommendation: build the object detection solution in-house.")
else:
    print("Model below target accuracy. Recommendation: consider a pre-trained or commercial solution.")
