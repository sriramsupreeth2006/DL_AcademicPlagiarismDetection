import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score
)

from model import SiamesePlagiarismModel


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Dataset class
class PairDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        return (
            str(row["s1_clean"]),
            str(row["s2_clean"]),
            int(row["Quality"])
        )


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Data loading function
def get_loader(file):

    dataset = PairDataset(file)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    return loader


# Load trained model
model = SiamesePlagiarismModel().to(DEVICE)

model.load_state_dict(
    torch.load("saved_model.pt", map_location=DEVICE)
)

model.eval()


# Prediction function
def get_predictions(loader):

    actual = []
    probabilities = []

    with torch.no_grad():

        for s1, s2, labels in loader:

            enc_a = tokenizer(
                list(s1),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            enc_b = tokenizer(
                list(s2),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            outputs = model(
                enc_a["input_ids"].to(DEVICE),
                enc_a["attention_mask"].to(DEVICE),
                enc_b["input_ids"].to(DEVICE),
                enc_b["attention_mask"].to(DEVICE)
            )

            probs = torch.softmax(outputs, dim=1)[:, 1]

            actual.extend(labels.tolist())

            probabilities.extend(probs.cpu().tolist())

    return actual, probabilities


# STEP 1: Load validation dataset
print("\nLoading validation dataset...")

val_loader = get_loader("val_clean.csv")

val_actual, val_probabilities = get_predictions(val_loader)


# STEP 2: Find best threshold using validation data
best_threshold = 0.5
best_f1 = 0

print("\nTesting different thresholds...\n")

for threshold in [x / 100 for x in range(30, 71)]:

    val_predicted = [
        1 if p >= threshold else 0
        for p in val_probabilities
    ]

    score = f1_score(
        val_actual,
        val_predicted,
        average="macro",
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Macro F1: {score:.4f}"
    )

    if score > best_f1:

        best_f1 = score
        best_threshold = threshold


print("\nBest Threshold:", best_threshold)

print("Best Validation Macro F1:", round(best_f1, 4))


# STEP 3: Evaluate on test dataset
print("\nLoading test dataset...")

test_loader = get_loader("test_clean.csv")

test_actual, test_probabilities = get_predictions(test_loader)


test_predicted = [
    1 if p >= best_threshold else 0
    for p in test_probabilities
]


# STEP 4: Display evaluation results
print("\nFinal Test Classification Report:\n")

print(
    classification_report(
        test_actual,
        test_predicted,
        labels=[0, 1],
        target_names=[
            "Not Paraphrase",
            "Paraphrase"
        ],
        zero_division=0
    )
)


accuracy = accuracy_score(
    test_actual,
    test_predicted
)

print("Test Accuracy:", round(accuracy, 4))

print("Selected Threshold:", best_threshold)


# STEP 5: Generate confusion matrix
cm = confusion_matrix(
    test_actual,
    test_predicted,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Paraphrase",
        "Paraphrase"
    ]
)

display.plot(cmap="Blues")

plt.title("Confusion Matrix - Optimized Threshold")

plt.savefig("confusion_matrix.png")

plt.show()

plt.close()

print("\nEvaluation Completed!")