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

### take the input file and itterate through the rows
answers="what if i change this? because it is."

# tokenize the text and calculate token IDs for answer_start and answer_end
maxlen=512 # 512 maximum number of tokens
# Tokenize the context, question, and answer
#tokenized_context = tokenizer.encode_plus(context, add_special_tokens=True, truncation=True, max_length=maxlen, padding='max_length')
#input_ids_context = tokenized_context['input_ids']
#tokenized_question = tokenizer.encode_plus(question, add_special_tokens=True, truncation=True, max_length=maxlen, padding='max_length')
#input_ids_question = tokenized_question['input_ids']
tokenized_answers = tokenizer.encode_plus(answers, add_special_tokens=True, truncation=True, max_length=maxlen, padding=False)
input_ids_answers = tokenized_answers['input_ids']
# Find the token IDs for answer_start and answer_end positions
answer_start_token = input_ids_answers[1] #finds the first token of the answer after the CLS token
answer_end_token = input_ids_answers[-2] #finds the last token of the answer before the CLS token

print(answer_start_token)
print(answer_end_token)




'''
# Process each line (excluding the header)
for row in reader[0:]:
    # Split the line into columns
    columns = reader.strip().split(',')

    # Extract the values from each row
    context = columns[0]
    question = columns[1]
    answers = columns[3]
    answer_start = columns[4]
    answer_end = columns[5]

    # Tokenize the data and calculate token IDs
    tokenized_data = tokenize_data(context, question, answers, answer_start, answer_end)

    # Access the token IDs for answer_start and answer_end
    answer_start_token_id = tokenized_data['input_ids'][tokenized_data['answer_start_token']]
    answer_end_token_id = tokenized_data['input_ids'][tokenized_data['answer_end_token']]

    # Update the answer_start and answer_end columns with token IDs
    columns[4] = str(answer_start_token_id)
    columns[5] = str(answer_end_token_id)

    # Join the updated columns
    updated_line = ','.join(columns)

    # Print or save the updated line as needed
    print(updated_line)
'''