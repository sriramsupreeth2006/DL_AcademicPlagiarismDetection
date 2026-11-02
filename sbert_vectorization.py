import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load preprocessed datasets
train = pd.read_csv("train_clean.csv")
val = pd.read_csv("val_clean.csv")
test = pd.read_csv("test_clean.csv")

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert sentences into embeddings
print("Generating Sentence-BERT embeddings...")

X_train_s1 = model.encode(train["s1_clean"].tolist(), show_progress_bar=True)
X_train_s2 = model.encode(train["s2_clean"].tolist(), show_progress_bar=True)

X_val_s1 = model.encode(val["s1_clean"].tolist(), show_progress_bar=True)
X_val_s2 = model.encode(val["s2_clean"].tolist(), show_progress_bar=True)

X_test_s1 = model.encode(test["s1_clean"].tolist(), show_progress_bar=True)
X_test_s2 = model.encode(test["s2_clean"].tolist(), show_progress_bar=True)

# Save embeddings
np.save("train_s1_sbert.npy", X_train_s1)
np.save("train_s2_sbert.npy", X_train_s2)

np.save("val_s1_sbert.npy", X_val_s1)
np.save("val_s2_sbert.npy", X_val_s2)

np.save("test_s1_sbert.npy", X_test_s1)
np.save("test_s2_sbert.npy", X_test_s2)

print("\nSentence-BERT Vectorization Completed!")

print("Training Shape:", X_train_s1.shape)
print("Validation Shape:", X_val_s1.shape)
print("Testing Shape:", X_test_s1.shape)