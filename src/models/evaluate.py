import torch
from transformers import DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from src.data.tokenize import tokenize_data

def evaluate_model(model_path="models/sentiment_model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    _, _, test_loader, _ = tokenize_data()  
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy}")
    
    print(classification_report(true_labels, predictions, labels=[0, 1], target_names=["Negative", "Positive"]))
    return predictions, true_labels

if __name__ == "__main__":
    predictions, true_labels = evaluate_model()