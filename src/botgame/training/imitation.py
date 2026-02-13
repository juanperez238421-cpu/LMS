import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
from typing import Tuple

from botgame.training.dataset import BotDataset

class ImitationPolicy(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for imitation learning.
    Takes observation features as input and outputs action targets.
    """
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

def train_imitation(data_dir: str, model_save_path: str, epochs: int = 10, batch_size: int = 32) -> None:
    """
    Trains an imitation learning policy using behavior cloning.
    Args:
        data_dir: Directory containing recorded episode data (jsonl.gz files).
        model_save_path: Path to save the trained model.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
    """
    dataset = BotDataset(data_dir)
    if len(dataset) == 0:
        print("No data available for imitation learning. Please generate data first.")
        return

    # Determine observation and action dimensions from the first sample
    sample_obs, sample_action = dataset[0]
    obs_dim = sample_obs.shape[0]
    action_dim = sample_action.shape[0]

    model = ImitationPolicy(obs_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss() # Mean Squared Error for regression tasks

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Starting imitation training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_action in dataloader:
            optimizer.zero_grad()
            predicted_action = model(batch_obs)
            loss = criterion(predicted_action, batch_action)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained imitation model saved to {model_save_path}")

def main():
    # Example usage:
    # Ensure you have run matches and recorded data in "data/processed"
    train_imitation(
        data_dir="data/processed",
        model_save_path="artifacts/imitation_policy.pt",
        epochs=50
    )

if __name__ == "__main__":
    main()
