#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: Generates the SQuAD json traing file from a human generated csv (which came from a google sheet)
### ### ### creates a dataset using text from:
# https://www.irs.gov/faqs 
# https://en.wikipedia.org/wiki/Taxation_in_the_United_States 
### csv file format
# column 0: context
# column 1: question
# column 2: question_id
# column 3: Answers
# column 4: answer_start
# column 5: answer_end

### ### ### Import necessary Libraries
import csv
import transformers
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

### input/output files
input_file = 'db/fine-tuning dataset - Sheet1.csv'
output_file = 'db/updated_csv.csv'

### BERT model for fine-tuning
model_name='bert-large-cased-whole-word-masking-finetuned-squad'
from transformers import BertTokenizer
# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
maxlen=512 # 512 maximum number of tokens
### take the input file and itterate through the rows

# Open the input CSV file
with open(input_file, 'r') as input_file:
    # Open the output CSV file
    with open(output_file, 'w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

         # Read and ignore the first row (headers)
        headers = next(reader)
        # Write the headers to the output file
        writer.writerow(headers)

        # Iterate over each row in the input file
        for row in reader:
            # Read the value from column 3
            answers = row[3]
            # tokenize the text and calculate token IDs for answer_start and answer_end
            # Tokenize the context, question, and answer
            #tokenized_context = tokenizer.encode_plus(context, add_special_tokens=True, truncation=True, max_length=maxlen, padding='max_length')
            #input_ids_context = tokenized_context['input_ids']
            #tokenized_question = tokenizer.encode_plus(question, add_special_tokens=True, truncation=True, max_length=maxlen, padding='max_length')
            #input_ids_question = tokenized_question['input_ids']
            tokenized_answers = tokenizer.encode_plus(answers, add_special_tokens=True, truncation=True, max_length=maxlen, padding=False)
            input_ids_answers = tokenized_answers['input_ids']
            # Find the token IDs for answer_start and answer_end positions
            answer_start_token = input_ids_answers[1] #finds the first token of the answer after the CLS token
            answer_end_token = input_ids_answers[-3] #finds the last token of the answer before the CLS token, and goes back one to avoid punctuation
            # Write the value to column 4
            row[4] = answer_start_token
            row[5] = answer_end_token
            # Write the updated row to the output file
            writer.writerow(row)