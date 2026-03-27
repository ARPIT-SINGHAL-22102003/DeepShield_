import torch
import torch.nn as nn
import librosa
import numpy as np
import os

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

def extract_melspectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, duration=4.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def predict_audio(audio_path):
    model = LCNN()
    
    if os.path.exists('weights/lcnn.pth'):
        model.load_state_dict(torch.load('weights/lcnn.pth', map_location='cpu'))
        print("LCNN weights loaded!")
    else:
        print("LCNN random weights use ho rahe hain!")
    
    model.eval()
    
    mel = extract_melspectrogram(audio_path)
    mel = librosa.util.fix_length(mel, size=400, axis=1)
    tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        score = model(tensor).item()
    
    return score