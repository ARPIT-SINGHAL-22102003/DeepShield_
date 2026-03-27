import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

# MesoNet Architecture
class MesoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 16 * 16, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.sigmoid(self.fc2(x))

def train():
    # Dataset path
    train_dir = r'C:\Users\ARPIT\deepshield\dataset_raw\real_vs_fake\real-vs-fake\train'
    valid_dir = r'C:\Users\ARPIT\deepshield\dataset_raw\real_vs_fake\real-vs-fake\valid'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                           [0.5, 0.5, 0.5])
    ])

    # Dataset load karo
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform
    )
    valid_dataset = datasets.ImageFolder(
        root=valid_dir,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False
    )

    print(f"✅ Training images: {len(train_dataset)}")
    print(f"✅ Validation images: {len(valid_dataset)}")
    print(f"✅ Classes: {train_dataset.classes}")

    # Model, loss, optimizer
    model = MesoNet()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0.0

    # Training loop - 10 epochs
    for epoch in range(10):
        # ---- Training ----
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Har 20 batches pe progress dikhao
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/10 | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Train Acc: {100*correct/total:.1f}%")

        train_accuracy = 100 * correct / total

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                labels = labels.float().unsqueeze(1)
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total

        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/10 Complete!")
        print(f"Train Accuracy : {train_accuracy:.1f}%")
        print(f"Val Accuracy   : {val_accuracy:.1f}%")
        print(f"{'='*50}\n")

        # Best model save karo
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), 'weights/mesonet.pth')
            print(f"💾 Best model saved! Val Accuracy: {val_accuracy:.1f}%\n")

    print(f"🎉 Training complete! Best accuracy: {best_accuracy:.1f}%")

if __name__ == "__main__":
    train()