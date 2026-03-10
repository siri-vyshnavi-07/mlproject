# src/model_tensorflow.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def build_tf_model(num_users, num_movies, embedding_dim=50):
    user_input  = keras.Input(shape=(1,), name="user_input")
    movie_input = keras.Input(shape=(1,), name="movie_input")

    user_vec  = keras.layers.Embedding(num_users,  embedding_dim, name="user_embedding")(user_input)
    movie_vec = keras.layers.Embedding(num_movies, embedding_dim, name="movie_embedding")(movie_input)

    user_vec  = keras.layers.Flatten()(user_vec)
    movie_vec = keras.layers.Flatten()(movie_vec)

    dot = keras.layers.Dot(axes=1)([user_vec, movie_vec])
    output = keras.layers.Dense(1, activation="linear")(dot)

    model = keras.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_tf_model(train_df, test_df, epochs=10, batch_size=256):
    num_users  = max(train_df["user_id"].max(), test_df["user_id"].max()) + 1
    num_movies = max(train_df["movie_id"].max(), test_df["movie_id"].max()) + 1

    model = build_tf_model(num_users, num_movies)
    model.summary()

    X_train = [train_df["user_id"].values, train_df["movie_id"].values]
    y_train = train_df["rating"].values.astype(np.float32)

    X_test  = [test_df["user_id"].values, test_df["movie_id"].values]
    y_test  = test_df["rating"].values.astype(np.float32)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    preds = model.predict(X_test, verbose=0).flatten()
    rmse  = np.sqrt(np.mean((preds - y_test) ** 2))
    print(f"\nTensorFlow Final RMSE: {rmse:.4f}")

    model.save("data/tf_model.keras")
    print("TensorFlow model saved!")
    return model, history, rmse


def get_recommendations_tf(model, user_id, movies_df, num_movies, top_k=10):
    all_movie_ids = np.arange(num_movies)
    user_array    = np.array([user_id] * num_movies)

    scores = model.predict([user_array, all_movie_ids], verbose=0).flatten()
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
    model, history, rmse = train_tf_model(train, test, epochs=10)
    print(f"\nFinal TensorFlow RMSE: {rmse:.4f}")