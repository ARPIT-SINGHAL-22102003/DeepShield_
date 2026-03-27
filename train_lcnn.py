import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os

# LCNN Architecture
class LCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 20 * 100, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

# Custom Audio Dataset
class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []
        
        # Real files = 0, Fake files = 1
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        for f in os.listdir(real_dir):
            if f.endswith('.wav') or f.endswith('.mp3'):
                self.files.append(os.path.join(real_dir, f))
                self.labels.append(0)
        
        for f in os.listdir(fake_dir):
            if f.endswith('.wav') or f.endswith('.mp3'):
                self.files.append(os.path.join(fake_dir, f))
                self.labels.append(1)
        
        print(f"Total files: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            y, sr = librosa.load(self.files[idx], sr=16000, duration=4.0)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = librosa.util.fix_length(mel_db, size=400, axis=1)
            tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
            return tensor, torch.tensor(self.labels[idx], dtype=torch.float32)
        except:
            # Error aaye toh blank return karo
            return torch.zeros(1, 80, 400), torch.tensor(0.0)

def train():
    train_dir = r'C:\Users\ARPIT\deepshield\audio_dataset\train'
    val_dir = r'C:\Users\ARPIT\deepshield\audio_dataset\val'

    print("Loading dataset...")
    train_dataset = AudioDataset(train_dir)
    val_dataset = AudioDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = LCNN()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0.0

    for epoch in range(10):
        # Training
        model.train()
        correct = 0
        total = 0

        for batch_idx, (specs, labels) in enumerate(train_loader):
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/10 | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {100*correct/total:.1f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for specs, labels in val_loader:
                labels = labels.unsqueeze(1)
                outputs = model(specs)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        train_acc = 100 * correct / total

        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/10 Complete!")
        print(f"Train Accuracy : {train_acc:.1f}%")
        print(f"Val Accuracy   : {val_acc:.1f}%")
        print(f"{'='*50}\n")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), 'weights/lcnn.pth')
            print(f"Best model saved! Val Accuracy: {val_acc:.1f}%\n")

    print(f"Training complete! Best accuracy: {best_accuracy:.1f}%")

if __name__ == "__main__":
    train()