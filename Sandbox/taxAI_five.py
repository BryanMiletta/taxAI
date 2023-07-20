#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Sandbox
#summary:this code uses pytorch with BERT 

### ### ### Import Libraries
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import AdamW
from torch.utils.data import DataLoader
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV2Processor
from transformers.data.metrics.squad_metrics import compute_predictions_logits

# Load the fine-tuned BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Set the device for running the model (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the SQuAD processor for data conversion
squad_processor = SquadV2Processor()

# Load the SQuAD training dataset
train_dataset = squad_processor.get_train_examples('db')
train_features, train_dataset = squad_convert_examples_to_features(
    examples=train_dataset,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=True,
    return_dataset="pt"
)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Fine-tune the BERT model with SQuAD training dataset
optimizer = AdamW(model.parameters(), lr=5e-5)
for epoch in range(3):  # Example: Run 3 epochs
    model.train()
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'start_positions': batch[3], 'end_positions': batch[4]}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Load the SQuAD evaluation dataset
eval_dataset = squad_processor.get_dev_examples('db')
eval_features, eval_dataset = squad_convert_examples_to_features(
    examples=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    return_dataset="pt"
)
eval_dataloader = DataLoader(eval_dataset, batch_size=16)

# Evaluate the fine-tuned BERT model on SQuAD evaluation dataset
model.eval()
all_results = []
for batch in eval_dataloader:
    batch = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
    with torch.no_grad():
        outputs = model(**inputs)
    for i in range(len(batch[0])):
        result = SquadResult(unique_id=batch[3][i].item(), start_logits=outputs.start_logits[i], end_logits=outputs.end_logits[i])
        all_results.append(result)

# Convert model output to predictions
predictions = compute_predictions_logits(
    eval_dataset.examples,
    eval_features,
    all_results,
    20,  # n_best_size, example: return top 20 predictions
    30,  # max_answer_length, example: limit answer length to 30 tokens
    True,  # do_lower_case, example: use lower case
    'db',  # Path to SQuAD eval dataset for evaluating exact match and f1 score
    'db/temp'  # Temporary directory for writing intermediate files
)

# Print predictions
for key in predictions.keys():
    print('Question:', key)
    print('Answer:', predictions[key])

# Save the fine-tuned BERT model
model.save_pretrained('db/temp/model')
tokenizer.save_pretrained('db/temp/model')
