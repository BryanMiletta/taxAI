
#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: Main run file to execute the model. UI to collect question, build dataset from 1040, run the data through the pre-training model. TODO run through fine-tuning model. Output result.

### ### ### Import necessary Libraries
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import torch.cuda as cuda

model_path = "db"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

device = torch.device('cuda' if cuda.is_available() else 'cpu')

import create_dataset
from loadModel import *
from run_fineTuning import *
import textwrap
import nltk
import PyPDF2
from PyPDF2 import PdfReader

### (Previous imports and code)

def handle_user_input(model, tokenizer):
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        context = input("Enter the context for your question: ")

        # Tokenize the input
        inputs = tokenizer(context, question, return_tensors="pt")

        # Move tensors to the appropriate device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get the model's predictions
        with torch.no_grad():
            outputs = model(**inputs)

        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)

        # Convert token indices to actual tokens and display the answer
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx+1]))
        print("Answer:", answer)

def fine_tune_and_save_model(train_dataset, val_dataset, tokenizer):
    args = TrainingArguments(
        f"test-squad",
        evaluation_strategy="epoch",
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

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("db")
    tokenizer.save_pretrained("db")

def main():
    # Access SQuAD fine-tuning datasets
    train_contexts, train_questions, train_answers = read_squad('db/json_file.json') 
    val_contexts, val_questions, val_answers = read_squad('db/Val.json')

    # Add index
    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Train encodings
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

    # Add token positional encodings
    add_token_positions(train_encodings, train_answers)
    add_token_positions(val_encodings, val_answers)
    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)

    # Initialize model
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)

    # Fine-tune and save the model
    fine_tune_and_save_model(train_dataset, val_dataset, tokenizer)

    # Handle user input and display model's answers
    model.eval()  # Set the model to evaluation mode
    handle_user_input(model, tokenizer)

if __name__ == "__main__":
    main()