"""Text classification by fine-tuning a Hugging Face Transformer.

Fine-tunes DistilBERT for sentiment/topic classification using a custom
PyTorch training loop with the Hugging Face tokenizer and model.

Run:  python text_classifier.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    DEVICE, TEXT_MODEL_NAME, TEXT_MODEL_PATH,
    TEXT_BATCH_SIZE, TEXT_EPOCHS, TEXT_LR, TEXT_MAX_LEN,
)


class TextDataset(Dataset):
    """Tokenizes raw text on the fly for the Transformer."""

    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=TEXT_MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def sample_data():
    """A small labeled sample (0 = negative, 1 = positive) for demonstration."""
    texts = [
        "This movie was fantastic and truly moving.",
        "Absolutely loved the performances and story.",
        "A brilliant, heartfelt film I'd watch again.",
        "Terrible plot and wooden acting throughout.",
        "I was bored and disappointed the whole time.",
        "A complete waste of two hours, avoid it.",
        "Wonderful direction and a gripping script.",
        "Dull, predictable, and poorly edited.",
    ]
    labels = [1, 1, 1, 0, 0, 0, 1, 0]
    return texts, labels


def train(epochs: int = TEXT_EPOCHS):
    """Fine-tune DistilBERT with a custom training loop."""
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        TEXT_MODEL_NAME, num_labels=2
    ).to(DEVICE)

    texts, labels = sample_data()
    dataset = TextDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=TEXT_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TEXT_LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            batch_labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_labels.size(0)
            correct += (outputs.logits.argmax(1) == batch_labels).sum().item()
            total += batch_labels.size(0)

        print(f"Epoch {epoch}/{epochs} | loss={total_loss/total:.3f} acc={correct/total:.3f}")

    torch.save(model.state_dict(), TEXT_MODEL_PATH)
    print(f"Saved fine-tuned model to {TEXT_MODEL_PATH}")
    return model, tokenizer


@torch.no_grad()
def predict(model, tokenizer, text: str) -> dict:
    """Predict the sentiment of a single piece of text."""
    model.eval()
    enc = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=TEXT_MAX_LEN, return_tensors="pt",
    ).to(DEVICE)
    logits = model(**enc).logits
    prob = torch.softmax(logits, dim=1)[0]
    label = int(prob.argmax())
    return {"label": "positive" if label == 1 else "negative",
            "confidence": round(float(prob[label]), 4)}


if __name__ == "__main__":
    model, tokenizer = train()
    demo = predict(model, tokenizer, "An emotional and beautifully made film.")
    print("Demo prediction:", demo)
