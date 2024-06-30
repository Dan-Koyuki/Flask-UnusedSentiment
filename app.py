from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask_cors import CORS

app = Flask(__name__)
CORS(app);

model_path = "./saved_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def preprocess_text(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    return input_ids, attention_mask


@app.route("/predict", methods=["POST"])
def predict_sentiment():
    if request.method == "POST":
        text = request.json.get("text", "")
        if text.strip() == "":
            return jsonify({"error": "Teks tidak boleh kosong"})

        input_ids, attention_mask = preprocess_text(text)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten().cpu().numpy()[0]

        sentiment = "positif" if preds == 1 else "negatif"
        return jsonify({"sentimen": sentiment})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
