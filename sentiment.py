from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json

# Inisialisasi tokenizer dan model
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p2")

def predict_sentiment(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]

    print(f"Text: {text}")  # Print text for debugging
    print(f"Probabilities: {probabilities}")  # Print probabilities for debugging

    if probabilities[1] > threshold:
        sentiment_label = "positif"
    else:
        sentiment_label = "negatif"

    return sentiment_label, probabilities

# Daftar kalimat untuk pengujian
test_sentences = [
    "Pelayanan di kantor desa sangat memuaskan.",
    "Proses pengurusan KTP di desa ini sangat cepat.",
    "Petugas desa sangat ramah dan membantu.",
    "Saya kecewa dengan pelayanan di kantor desa, sangat lambat.",
    "Sistem antrian di balai desa sangat tidak teratur."
]

# Lakukan prediksi sentimen untuk setiap kalimat
results = []
for text in test_sentences:
    label, probabilities = predict_sentiment(text, threshold=0.4)
    result = {
        "text": text,
        "sentiment_label": label,
        "sentiment_probabilities": {
            "positif": probabilities[1],
            "negatif": probabilities[0],
        },
    }
    results.append(result)

# Print output as JSON to stdout
print(json.dumps(results, indent=4))
