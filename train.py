import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, classification_report

# Baca data dan hilangkan duplikat
df = pd.read_csv("data.csv")
df = df.drop_duplicates()

# Pisahkan teks dan label
texts = df["text"].values
labels = df["sentimen"].apply(lambda x: 1 if x == "positif" else 0).values

# Pisahkan data untuk pelatihan dan validasi
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenisasi teks
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
max_len = 128
train_encodings = tokenizer(
    train_texts.tolist(), truncation=True, padding=True, max_length=max_len
)
val_encodings = tokenizer(
    val_texts.tolist(), truncation=True, padding=True, max_length=max_len
)

train_inputs = torch.tensor(train_encodings["input_ids"])
train_masks = torch.tensor(train_encodings["attention_mask"])
train_labels = torch.tensor(train_labels)
val_inputs = torch.tensor(val_encodings["input_ids"])
val_masks = torch.tensor(val_encodings["attention_mask"])
val_labels = torch.tensor(val_labels)

# Buat dataset untuk pelatihan dan validasi
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

# DataLoader untuk pelatihan dan validasi
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load model BERT untuk klasifikasi
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Fungsi evaluasi model
def evaluate_model(model, val_dataloader):
    model.eval()
    val_loss, val_accuracy = 0, 0
    all_preds, all_labels = [], []

    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch

        with torch.no_grad():
            outputs = model(inputs, attention_mask=masks, labels=labels)

        logits = outputs.logits
        loss = outputs.loss
        val_loss += loss.item()

        preds = torch.argmax(logits, dim=1).flatten()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_dataloader)
    val_accuracy = accuracy_score(all_labels, all_preds)

    return val_loss, val_accuracy, all_preds, all_labels

# Pelatihan model
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch

        model.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}")

val_loss, val_accuracy, val_preds, val_labels = evaluate_model(model, val_dataloader)

print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
print(classification_report(val_labels, val_preds, target_names=["negatif", "positif"]))

print("\nContoh data dari data.csv:")
print(df.head())

model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
print("\nModel dan tokenizer telah disimpan di './saved_model'")

# Ekstraksi kata-kata unik dari dataset
unique_words = set()
for text in texts:
    for word in text.split():
        unique_words.add(word)

# Fungsi klasifikasi dengan pengecekan kata tidak dikenali
def classify_sentence(sentence):
    words = sentence.split()
    for word in words:
        if word not in unique_words:
            return "tidak_dikenali", 1.0

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(device)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, predicted_class = torch.max(probs, dim=1)
    classes = ["negatif", "positif"]
    return classes[predicted_class], confidence.item()

# Contoh penggunaan
sentence = "Contoh kalimat tidak dikenal"
label, confidence = classify_sentence(sentence)
print(f"Label: {label}, Confidence: {confidence}")
