import torch
from transformers import AutoTokenizer

from model import SiamesePlagiarismModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

model = SiamesePlagiarismModel().to(device)

model.load_state_dict(
    torch.load("saved_model.pt", map_location=device)
)

model.eval()


def predict(sentence1, sentence2):

    enc_a = tokenizer(
        sentence1,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    enc_b = tokenizer(
        sentence2,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            enc_a["input_ids"].to(device),
            enc_a["attention_mask"].to(device),
            enc_b["input_ids"].to(device),
            enc_b["attention_mask"].to(device)
        )

        probabilities = torch.softmax(outputs, dim=1)

        probability = probabilities[0][1].item()

        result = "Paraphrase" if probability >= 0.5 else "Not Paraphrase"

    return probability * 100, result


if __name__ == "__main__":

    sentence1 = input("Enter first sentence: ")

    sentence2 = input("Enter second sentence: ")

    score, result = predict(sentence1, sentence2)

    print("\nPrediction:", result)

    print("Paraphrase probability:", round(score, 2), "%")