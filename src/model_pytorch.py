# src/model_pytorch.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users  = torch.tensor(df["user_id"].values,  dtype=torch.long)
        self.movies = torch.tensor(df["movie_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values,  dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embedding  = nn.Embedding(num_users,  embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_ids, movie_ids):
        user_vec  = self.user_embedding(user_ids)
        movie_vec = self.movie_embedding(movie_ids)
        x = torch.cat([user_vec, movie_vec], dim=1)
        return self.fc_layers(x).squeeze()


def train_pytorch_model(train_df, test_df, epochs=10, lr=0.001, batch_size=256):
    num_users  = max(train_df["user_id"].max(), test_df["user_id"].max()) + 1
    num_movies = max(train_df["movie_id"].max(), test_df["movie_id"].max()) + 1

    train_loader = DataLoader(RatingsDataset(train_df), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(RatingsDataset(test_df),  batch_size=batch_size)

    model     = NCF(num_users, num_movies)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, test_rmses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, movies, ratings in train_loader:
            optimizer.zero_grad()
            preds = model(users, movies)
            loss  = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for users, movies, ratings in test_loader:
                preds = model(users, movies)
                all_preds.extend(preds.numpy())
                all_targets.extend(ratings.numpy())

        rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2))
        test_rmses.append(rmse)
        print(f"[PyTorch] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test RMSE: {rmse:.4f}")

    torch.save(model.state_dict(), "data/pytorch_model.pth")
    print("\nPyTorch model saved!")
    return model, train_losses, test_rmses


def get_recommendations_pytorch(model, user_id, movies_df, num_users, num_movies, top_k=10):
    model.eval()
    all_movie_ids = torch.arange(num_movies, dtype=torch.long)
    user_tensor   = torch.tensor([user_id] * num_movies, dtype=torch.long)

    with torch.no_grad():
        scores = model(user_tensor, all_movie_ids).numpy()

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        title = movies_df[movies_df["movie_id"] == idx + 1]["title"].values
        title = title[0] if len(title) > 0 else f"Movie {idx}"
        results.append({"movie_id": idx, "title": title, "predicted_rating": round(float(scores[idx]), 2)})

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import load_data, split_data
    df, movies = load_data()
    train, test = split_data(df)
    model, losses, rmses = train_pytorch_model(train, test, epochs=10)
    print(f"\nFinal PyTorch RMSE: {rmses[-1]:.4f}")