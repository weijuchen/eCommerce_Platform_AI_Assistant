# nlp_models.py
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class IntentClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]
        return self.classifier(cls_embedding)


class SentimentAnalyzer(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = IntentClassifier()
    inputs = tokenizer("我要退貨", return_tensors="pt")
    outputs = model(**inputs)
    print("✅ IntentClassifier 測試輸出:", outputs)
