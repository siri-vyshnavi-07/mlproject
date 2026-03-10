# 🎬 Movie Recommendation System

A movie recommendation system built with **PyTorch** and **TensorFlow**, trained on the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset.

## Models

| Model                                | Framework        | Technique                   |
| ------------------------------------ | ---------------- | --------------------------- |
| Neural Collaborative Filtering (NCF) | PyTorch          | Embeddings + Neural Network |
| Matrix Factorization (MF)            | TensorFlow/Keras | Embeddings + Dot Product    |

## Results

| Model         | Test RMSE |
| ------------- | --------- |
| PyTorch NCF   | ~0.95     |
| TensorFlow MF | ~0.92     |

## Quick Start

```bash
pip install -r requirements.txt
python src/data_loader.py
python src/model_pytorch.py
python src/model_tensorflow.py
streamlit run app.py
```

## Tech Stack

- **PyTorch** — Neural Collaborative Filtering
- **TensorFlow/Keras** — Matrix Factorization
- **Streamlit** — Interactive web demo
- **Pandas / NumPy / Scikit-learn** — Data processing
