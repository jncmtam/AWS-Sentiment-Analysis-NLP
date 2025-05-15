import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def predict_sentiment(text, model_path="models/sentiment_model", max_length=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    return "Negative" if pred == 0 else "Positive"

if __name__ == "__main__":
    sample_text = "This is a great product!"
    result = predict_sentiment(sample_text)
    print(f"Sentiment: {result}")