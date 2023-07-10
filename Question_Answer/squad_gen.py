#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: Generates the SQuAD json traing file from a human generated csv (which came from a google sheet)
# column 1: context
# column 2: question
# column 3: question_id
# column 4: Answers
# column 5: answer_start
# column 6: answer_end

### ### ### Import necessary Libraries
import csv

input_file = 'db/fine-tuning dataset - Sheet1.csv'
output_file = 'db/updated_csv.csv'

### ### ### creates a dataset using text from:
# https://www.irs.gov/faqs 
# https://en.wikipedia.org/wiki/Taxation_in_the_United_States 

from transformers import BertTokenizer

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Function to tokenize the text and calculate token IDs for answer_start and answer_end
def tokenize_data(text, question, answers, answer_start, answer_end):
    # Tokenize the context, question, and answer
    tokenized_input = tokenizer(text, question, truncation=True, padding=True)

    # Find the token IDs for answer_start and answer_end positions
    answer_start_token = tokenized_input.char_to_token(answer_start)
    answer_end_token = tokenized_input.char_to_token(answer_end)

    # Update the tokenized_input dictionary with token IDs
    tokenized_input['answer_start_token'] = answer_start_token
    tokenized_input['answer_end_token'] = answer_end_token

    return tokenized_input

# Open the CSV file and process each row
with open(input_file, 'r') as file:
    lines = file.readlines()

    # Process each line (excluding the header)
    for line in lines[1:]:
        # Split the line into columns
        columns = line.strip().split(',')

        # Extract the values from each column
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