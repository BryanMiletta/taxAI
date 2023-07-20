#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: run file for fine-tuning 

### ### ### Import necessary Libraries
from FineTune import *
from create_Squad_DS import *
import torch
import torch.cuda as cuda
from transformers import BertForQuestionAnswering, BertTokenizerFast
bert_model = 'bert-base-uncased'

device = torch.device('cuda' if cuda.is_available() else 'cpu')


### Access SQuAD fine-tuning datasets
train_contexts, train_questions, train_answers = read_squad('db/json_file.json') 
val_contexts, val_questions, val_answers = read_squad('db/Val.json') #TODO can change this to dev-2.0.json to improve performance.

# Add index
add_end_idx(train_answers, train_contexts) #TODO might need to correct end_idx setting
add_end_idx(val_answers, val_contexts) 

# import tokenizer

tokenizer = BertTokenizerFast.from_pretrained(bert_model)

# Train encodings
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

# add token positional encodings
add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)
train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

model = BertForQuestionAnswering.from_pretrained(bert_model).to(device)

# Optimizer
from torch.utils.data import DataLoader
from transformers import AdamW,TrainingArguments, Trainer,default_data_collator
args = TrainingArguments(
    f"test-squad",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
)
data_collator = default_data_collator
trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)
trainer.train()

### Train
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=0)

# optimize
optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.save_pretrained("db/model")
tokenizer.save_pretrained("db/model/tokenizer")
# evaluate results
model.eval()