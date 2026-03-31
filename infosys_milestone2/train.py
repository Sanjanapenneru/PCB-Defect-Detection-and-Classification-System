import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ======================
# 1. Setup & Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================
# 2. Advanced Transforms for Small Defects
# ======================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ======================
# 3. Data Loaders
# ======================
train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
val_dataset = datasets.ImageFolder("dataset/val", transform=val_transform)

# Batch size 16 is better for CPU stability
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Detected Classes: {class_names}")

# ======================
# 4. Model (EfficientNet-B0 Full Unfreeze)
# ======================
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# Unfreeze everything so the model can learn PCB-specific patterns
for param in model.parameters():
    param.requires_grad = True

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)
model = model.to(device)

# ======================
# 5. Optimized Optimizer (SGD + Momentum)
# ======================
criterion = nn.CrossEntropyLoss()
# SGD with momentum helps bypass the 16% "random guess" plateau
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

# This drops LR by 0.1 only when progress stops
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# ======================
# 6. Training Loop
# ======================
num_epochs = 25 # Increased epochs to give CPU time to reach 90%
best_acc = 0.0

print("\n--- Training Started ---")
for epoch in range(num_epochs):
    model.train()
    t_correct, t_total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        t_total += labels.size(0)
        t_correct += (predicted == labels).sum().item()

    train_acc = 100 * t_correct / t_total

    # Validation
    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            v_total += labels.size(0)
            v_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * v_correct / v_total
    
    # Update scheduler based on validation accuracy
    scheduler.step(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_pcb_model.pth")
        print("--> Best Model Saved!")

print(f"\nTraining Finished! Highest Accuracy: {best_acc:.2f}%")