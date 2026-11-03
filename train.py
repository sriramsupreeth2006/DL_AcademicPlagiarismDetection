import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from model import SiamesePlagiarismModel


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Dataset class
class PairDataset(Dataset):

    def __init__(self, file):
        self.df = pd.read_csv(file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        return (
            str(row["s1_clean"]),
            str(row["s2_clean"]),
            int(row["Quality"])
        )


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Tokenization function
def collate_fn(batch):

    s1, s2, labels = zip(*batch)

    token1 = tokenizer(
        list(s1),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    token2 = tokenizer(
        list(s2),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    return (
        token1["input_ids"],
        token1["attention_mask"],
        token2["input_ids"],
        token2["attention_mask"],
        torch.tensor(labels, dtype=torch.long)
    )


# Training function
def train():

    train_data = PairDataset("train_clean.csv")
    val_data = PairDataset("val_clean.csv")

    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize model
    model = SiamesePlagiarismModel().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5
    )

    # Calculate class weights
    class_counts = (
        train_data.df["Quality"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    class_weights = len(train_data) / (2 * class_counts)

    class_weights = torch.tensor(
        class_weights.values,
        dtype=torch.float32
    ).to(DEVICE)

    print("\nClass Weights:")
    print("Not Paraphrase:", class_weights[0].item())
    print("Paraphrase:", class_weights[1].item())

    # Weighted loss function
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    best_f1 = 0

    # Training epochs
    for epoch in range(3):

        model.train()

        total_loss = 0

        for ids1, mask1, ids2, mask2, labels in train_loader:

            ids1 = ids1.to(DEVICE)
            mask1 = mask1.to(DEVICE)

            ids2 = ids2.to(DEVICE)
            mask2 = mask2.to(DEVICE)

            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                ids1, mask1,
                ids2, mask2
            )

            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()

        actual = []
        predicted = []

        with torch.no_grad():

            for ids1, mask1, ids2, mask2, labels in val_loader:

                ids1 = ids1.to(DEVICE)
                mask1 = mask1.to(DEVICE)

                ids2 = ids2.to(DEVICE)
                mask2 = mask2.to(DEVICE)

                outputs = model(
                    ids1, mask1,
                    ids2, mask2
                )

                predictions = torch.argmax(
                    outputs, dim=1
                )

                actual.extend(labels.tolist())
                predicted.extend(predictions.cpu().tolist())

        # Calculate metrics for both classes
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual,
            predicted,
            average="macro",
            zero_division=0
        )

        accuracy = accuracy_score(actual, predicted)

        print("\nEpoch:", epoch + 1)

        print("Loss:", total_loss / len(train_loader))

        print("Accuracy:", accuracy)

        print("Macro Precision:", precision)

        print("Macro Recall:", recall)

        print("Macro F1 Score:", f1)

        # Save best model
        if f1 > best_f1:

            best_f1 = f1

            torch.save(
                model.state_dict(),
                "saved_model.pt"
            )

            print("Best model saved!")

    print("\nTraining Completed!")

    print("Best Validation Macro F1:", best_f1)


if __name__ == "__main__":
    train()