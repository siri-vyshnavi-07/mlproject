# app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import os, sys

sys.path.append(".")
from src.data_loader import load_data, split_data
from src.model_pytorch import NCF, get_recommendations_pytorch
from src.model_tensorflow import get_recommendations_tf

st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.markdown("Powered by **PyTorch (NCF)** and **TensorFlow (Matrix Factorization)**")
st.markdown("---")

@st.cache_data
def load_all_data():
    df, movies = load_data()
    train, test = split_data(df)
    return df, movies, train, test

df, movies, train, test = load_all_data()
num_users  = df["user_id"].nunique()
num_movies = df["movie_id"].nunique()

st.sidebar.header("Settings")
user_id      = st.sidebar.slider("Select User ID", 0, num_users - 1, 0)
top_k        = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
model_choice = st.sidebar.radio("Model", ["PyTorch (NCF)", "TensorFlow (MF)", "Compare Both"])

st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Ratings", f"{len(df):,}")
col2.metric("Users", num_users)
col3.metric("Movies", num_movies)
col4.metric("Avg Rating", f"{df['rating'].mean():.2f} ⭐")
st.markdown("---")

@st.cache_resource
def load_pytorch_model():
    if not os.path.exists("data/pytorch_model.pth"):
        st.warning("PyTorch model not found. Please run: python src/model_pytorch.py")
        return None
    model = NCF(num_users, num_movies)
    model.load_state_dict(torch.load("data/pytorch_model.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_tf_model():
    if not os.path.exists("data/tf_model.keras"):
        st.warning("TensorFlow model not found. Please run: python src/model_tensorflow.py")
        return None
    return tf.keras.models.load_model("data/tf_model.keras")

st.subheader(f"🎯 Recommendations for User {user_id}")

if model_choice in ["PyTorch (NCF)", "Compare Both"]:
    pt_model = load_pytorch_model()
    if pt_model:
        st.markdown("### 🔵 PyTorch — Neural Collaborative Filtering")
        recs_pt = get_recommendations_pytorch(pt_model, user_id, movies, num_users, num_movies, top_k)
        st.dataframe(recs_pt, use_container_width=True)

if model_choice in ["TensorFlow (MF)", "Compare Both"]:
    tf_model = load_tf_model()
    if tf_model:
        st.markdown("### 🟠 TensorFlow — Matrix Factorization")
        recs_tf = get_recommendations_tf(tf_model, user_id, movies, num_movies, top_k)
        st.dataframe(recs_tf, use_container_width=True)

st.markdown("---")
st.subheader(f"📋 Movies User {user_id} Has Rated")
user_history = df[df["user_id"] == user_id][["title", "rating"]].sort_values("rating", ascending=False)
st.dataframe(user_history.head(10), use_container_width=True)

st.markdown("---")
st.markdown("Built with ❤️ using MovieLens 100K | PyTorch | TensorFlow | Streamlit")