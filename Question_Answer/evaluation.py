#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: run file to execute the pre-training model. UI to collect question, build dataset from 1040, run the data through the pre-training model. TODO run through fine-tuning model. Output result.

### ### ### Import necessary Libraries
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering
from FineTune import read_squad, add_end_idx, add_token_positions
from create_Squad_DS import SquadDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

### ### ### PRE-TRAINING & Fine-Tuning
#model_path =  'db/model_finetune'
#tokenizer_path =  'db/model_finetune/tokenizer'
#model_path =  'db/model2'
#tokenizer_path =  'db/model2/tokenizer2'
model_path =  'bert-base-cased'
tokenizer_path =  'bert-base-cased'
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
### ### ### ###
# Step 1: Load the fine-tuned BERT model and tokenizer
# Step 2: Prepare the test dataset
test_contexts, test_questions, test_answers = read_squad('db/dev-v2.0.json')
add_end_idx(test_answers, test_contexts)
test_encodings = tokenizer(test_contexts, test_questions, truncation=True, padding=True)
add_token_positions(test_encodings, test_answers)
test_dataset = SquadDataset(test_encodings)
# Step 3: Use the model to make predictions on the test dataset
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, pin_memory=True, num_workers=0)
predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        _, predicted_starts = start_logits.max(1)
        _, predicted_ends = end_logits.max(1)
        predicted_starts = predicted_starts.cpu().tolist()
        predicted_ends = predicted_ends.cpu().tolist()
        predictions.extend(zip(predicted_starts, predicted_ends))

# Step 4: Evaluate the predictions against the ground truth labels
true_starts = [example["start_positions"] for example in test_dataset]
true_ends = [example["end_positions"] for example in test_dataset]

# Flatten the lists of tuples to create 1D arrays
predicted_starts, predicted_ends = zip(*predictions)

accuracy = accuracy_score(true_starts, predicted_starts)
f1 = f1_score(true_starts, predicted_starts, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")