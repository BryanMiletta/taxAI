#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Sandbox
#summary: Basic BERT

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Sample text for classification
text = "This is a sample sentence."

# Tokenize the input text
tokens = tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, max_length=512, padding='max_length')
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# Convert input to tensors
input_ids = torch.tensor([input_ids])
attention_mask = torch.tensor([attention_mask])

# Forward pass through the model
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Get the predicted probabilities
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
predicted_class = torch.argmax(probs)

# Print the predicted class
print("Predicted class:", predicted_class.item())
