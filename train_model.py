import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import numpy as np

class TrainModel:
    def __init__(self, train_file="train_dataset.pt",test_file='test_dataset.pt'):
        self.train_data = torch.load(train_file)
        self.test_data = torch.load(test_file)

        combined_labels = []
        combined_labels_test=[]

        for _, status, emotion, _ in self.train_data:
            combined_labels.append(f"{status.lower()}")

        for _, status, emotion, _ in self.test_data:
            combined_labels_test.append(f"{status.lower()}")
        #   To check if the test and train were properly suffled based on the SP/HP
        print('combined', combined_labels)
        print('test', combined_labels_test)

        self.labels = sorted(set(combined_labels))
        self.label_map = {l: i for i, l in enumerate(self.labels)}

        self.num_classes = len(self.label_map)

        print(self.num_classes)
        print(self.labels)
        print(self.label_map)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN feature extractor
        cnn = resnet18(weights="DEFAULT")
        self.feature_extractor = nn.Sequential(*list(cnn.children())[:-1])  # output: (B, 512, 1, 1)
        self.feature_extractor.to(self.device).eval()  # freeze CNN
        for p in self.feature_extractor.parameters(): p.requires_grad = False

        # LSTM classifier
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.classifier = nn.Linear(256, self.num_classes)

        self.lstm.to(self.device)
        self.classifier.to(self.device)

        self.optimizer = torch.optim.Adam(list(self.lstm.parameters()) + list(self.classifier.parameters()), lr=1e-4)

    class VideoDataset(Dataset):
        def __init__(self, data, label_map):
            self.data = data
            self.label_map = label_map

        def __len__(self): return len(self.data)

        def __getitem__(self, idx):
            _, status, emotion, frames = self.data[idx]
            combined_label = f"{status.lower()}"
            frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)  # (T, C, H, W)
            label = torch.tensor(self.label_map[combined_label], dtype=torch.long)
            return frames, label

    def _get_features(self, batch_videos):
        # batch_videos: (B, T, C, H, W)
        B, T, C, H, W = batch_videos.shape
        videos = batch_videos.view(B*T, C, H, W).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(videos).squeeze(-1).squeeze(-1)  # (B*T, 512)
        return feats.view(B, T, -1)  # (B, T, 512)

    def train_model(self, epochs=5, batch_size=4):
        print("Training started")
        train_loader = DataLoader(self.VideoDataset(self.train_data, self.label_map),
                                  batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total, correct, loss_sum = 0, 0, 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x, y = x.to(self.device), y.to(self.device)
                feats = self._get_features(x)  # (B, T, 512)
                lstm_out, _ = self.lstm(feats)  # (B, T, 256)
                final_hidden = lstm_out[:, -1, :]  # (B, 256)
                out = self.classifier(final_hidden)  # (B, num_classes)

                loss = F.cross_entropy(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

            print(f"Epoch {epoch+1}: Loss={loss_sum:.4f}, Acc={correct/total:.4f}")
        print("Training complete.")

    def make_predictions(self):
        print("Evaluating on test set...")
        test_loader = DataLoader(self.VideoDataset(self.test_data, self.label_map), batch_size=4)
        total, correct = 0, 0
        predictions = []

        self.lstm.eval()
        self.classifier.eval()

        with torch.no_grad():
            for x, y in tqdm(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                feats = self._get_features(x)
                lstm_out, _ = self.lstm(feats)
                out = self.classifier(lstm_out[:, -1, :])
                pred = out.argmax(1)
                predictions.extend(pred.cpu().tolist())
                correct += (pred == y).sum().item()
                total += y.size(0)

        print(f"Test Accuracy: {correct / total:.4f}")
        return predictions