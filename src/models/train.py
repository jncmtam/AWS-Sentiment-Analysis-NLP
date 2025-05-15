import torch
from transformers import DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from src.data.tokenize import tokenize_data
from src.models.config import config
from tqdm import tqdm

def train_model():
    # Early stopping
    
    train_loader, val_loader, _, tokenizer = tokenize_data()
    
    
    model = DistilBertForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=config.NUM_LABELS).to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_loss = float("inf")
    patience = config.PATIENCE
    epochs_no_improve = 0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")

        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                labels = batch["label"].to(config.DEVICE)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model.save_pretrained(config.MODEL_PATH)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    train_model()