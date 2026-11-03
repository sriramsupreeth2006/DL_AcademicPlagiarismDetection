import torch
import torch.nn as nn
from transformers import AutoModel


class SiamesePlagiarismModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(

            nn.Linear(hidden_size * 4, 256),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 2)
        )

    def encode(self, input_ids, attention_mask):

        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        embeddings = output.last_hidden_state

        mask = attention_mask.unsqueeze(-1).expand(
            embeddings.size()
        ).float()

        summed = torch.sum(embeddings * mask, dim=1)

        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        return summed / counts

    def forward(self, ids_a, mask_a, ids_b, mask_b):

        emb_a = self.encode(ids_a, mask_a)

        emb_b = self.encode(ids_b, mask_b)

        features = torch.cat(
            [
                emb_a,
                emb_b,
                torch.abs(emb_a - emb_b),
                emb_a * emb_b
            ],
            dim=1
        )

        output = self.classifier(features)

        return output