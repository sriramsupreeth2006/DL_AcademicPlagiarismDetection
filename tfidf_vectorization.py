import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import joblib

# Load preprocessed datasets
train = pd.read_csv("train_clean.csv")
val = pd.read_csv("val_clean.csv")
test = pd.read_csv("test_clean.csv")

# Initialize TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

# Fit only on training sentences
train_text = pd.concat([
    train["s1_clean"],
    train["s2_clean"]
])

tfidf.fit(train_text)

# Convert sentences into TF-IDF vectors
X_train_s1 = tfidf.transform(train["s1_clean"])
X_train_s2 = tfidf.transform(train["s2_clean"])

X_val_s1 = tfidf.transform(val["s1_clean"])
X_val_s2 = tfidf.transform(val["s2_clean"])

X_test_s1 = tfidf.transform(test["s1_clean"])
X_test_s2 = tfidf.transform(test["s2_clean"])

# Save vectorized data
save_npz("train_s1_tfidf.npz", X_train_s1)
save_npz("train_s2_tfidf.npz", X_train_s2)

save_npz("val_s1_tfidf.npz", X_val_s1)
save_npz("val_s2_tfidf.npz", X_val_s2)

save_npz("test_s1_tfidf.npz", X_test_s1)
save_npz("test_s2_tfidf.npz", X_test_s2)

# Save trained vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("TF-IDF Vectorization Completed!")

print("Training Shape:", X_train_s1.shape)
print("Validation Shape:", X_val_s1.shape)
print("Testing Shape:", X_test_s1.shape)