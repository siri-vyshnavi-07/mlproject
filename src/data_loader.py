# src/data_loader.py
# Downloads and prepares the MovieLens 100K dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile
import os

def download_data():
    """Downloads MovieLens 100K dataset automatically."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/u.data"):
        print("Downloading MovieLens 100K dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        urllib.request.urlretrieve(url, "data/ml-100k.zip")
        with zipfile.ZipFile("data/ml-100k.zip", "r") as z:
            z.extractall("data/")
        import shutil
        for f in os.listdir("data/ml-100k"):
            shutil.copy(f"data/ml-100k/{f}", f"data/{f}")
        print("Download complete!")
    else:
        print("Dataset already exists, skipping download.")

def load_data():
    download_data()
    ratings = pd.read_csv(
        "data/u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    movies = pd.read_csv(
        "data/u.item",
        sep="|",
        encoding="latin-1",
        usecols=[0, 1],
        names=["movie_id", "title"]
    )
    df = ratings.merge(movies, on="movie_id")
    df["user_id"] = pd.Categorical(df["user_id"]).codes
    df["movie_id"] = pd.Categorical(df["movie_id"]).codes
    print(f"Loaded {len(df)} ratings | {df['user_id'].nunique()} users | {df['movie_id'].nunique()} movies")
    return df, movies

def split_data(df, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    print(f"Train size: {len(train)} | Test size: {len(test)}")
    return train, test

def get_stats(df):
    return {
        "num_users": df["user_id"].nunique(),
        "num_movies": df["movie_id"].nunique(),
        "num_ratings": len(df),
        "avg_rating": round(df["rating"].mean(), 2),
        "sparsity": round(1 - len(df) / (df["user_id"].nunique() * df["movie_id"].nunique()), 4)
    }

if __name__ == "__main__":
    df, movies = load_data()
    train, test = split_data(df)
    stats = get_stats(df)
    print("\nDataset Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")