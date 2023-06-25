#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Sandbox
#summary:this code is testing setting up a Q/A BERT model that allows user input. 



### ### ### Import Libraries
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV2Processor
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate

# Load the BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load the SQuAD 2.0 dataset
train_file = "db"
dev_file = "db"

processor = SquadV2Processor()
train_examples = processor.get_train_examples(train_file)
dev_examples = processor.get_dev_examples(dev_file)

# Convert examples to features
train_features, train_dataset = squad_convert_examples_to_features(
    examples=train_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=True,
    return_dataset="pt",
    threads=1,
)

dev_features, dev_dataset = squad_convert_examples_to_features(
    examples=dev_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    return_dataset="pt",
    threads=1,
)

# Fine-tune the BERT model on the SQuAD 2.0 dataset
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 2
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Average training loss: {avg_train_loss}")

# Evaluate the fine-tuned model
model.eval()
all_predictions = []
all_true_labels = []

for batch in dev_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predictions = compute_predictions_logits(
        batch,
        outputs.start_logits,
        outputs.end_logits,
        tokenizer,
        20,
        30,
        True
    )

    all_predictions.extend(predictions)
    all_true_labels.extend(batch['answers'])

# Evaluate the predictions
dev_metrics = squad_evaluate(examples=dev_examples, preds=all_predictions, no_answer_probs=None)
print("Evaluation metrics:", dev_metrics)

# Example usage
context = "The quick brown fox jumps over the lazy dog."
question = "What does the fox jump over?"
answer = answer_question(context, question)
print(answer)