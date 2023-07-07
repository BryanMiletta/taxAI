#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: run file for fine-tuning 

### ### ### Import necessary Libraries
from FineTune import *
from create_Squad_DS import *

### Access SQuAD fine-tuning datasets
train_contexts, train_questions, train_answers = read_squad('squad_Sample.json') #TODO
val_contexts, val_questions, val_answers = read_squad('Val.json') #TODO 

# Add index
add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

# import tokenizer
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Train encodings
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

# add token positional encodings
add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)
train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

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
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# optimize
optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()

# evaluate results
model.eval()