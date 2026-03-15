# Movie Recommendation System

A movie recommendation system built with **PyTorch** and **TensorFlow**, 
trained on the MovieLens 100K dataset. Similar to how Netflix recommends 
what to watch next!

## What it does
- Takes a user ID as input
- Predicts which movies that user will like
- Returns top 10 movie recommendations instantly
- Compares two different ML models side by side

## Dataset
- **Source:** MovieLens 100K
- **Ratings:** 100,000 real movie ratings
- **Users:** 943
- **Movies:** 1,682
- **Average Rating:** 3.53 / 5

## Models
| Model | Framework | Technique | RMSE |
|-------|-----------|-----------|------|
| Neural Collaborative Filtering | PyTorch | Embeddings + Neural Network | 0.9556 |
| Matrix Factorization | TensorFlow | Embeddings + Dot Product | 0.9196 |

## Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/siri-vyshnavi-07/mlproject.git
cd mlproject

# 2. Create virtual environment (Python 3.11 required)
py -3.11 -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset and train models
python src/data_loader.py
python src/model_pytorch.py
python src/model_tensorflow.py

# 5. Launch the app
streamlit run src/app.py
```

## Tech Stack
- **PyTorch** — Neural Collaborative Filtering model
- **TensorFlow/Keras** — Matrix Factorization model
- **Streamlit** — Interactive web demo
- **Pandas & NumPy** — Data processing
- **Scikit-learn** — Train/test splitting

## Project Structure
```
mlproject/
├── src/
│   ├── data_loader.py       # Downloads & prepares MovieLens dataset
│   ├── model_pytorch.py     # PyTorch NCF model
│   ├── model_tensorflow.py  # TensorFlow MF model
│   └── app.py               # Streamlit demo app
├── requirements.txt
└── README.md
```
