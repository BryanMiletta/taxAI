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
def tokenize_data(context, question, answers, answer_start, answer_end):
    # Tokenize the context, question, and answer
    tokenized_input = tokenizer(context, question, truncation=True, padding=True)

    # Find the token IDs for answer_start and answer_end positions
    answer_start_token = tokenized_input.char_to_token(answer_start)
    answer_end_token = tokenized_input.char_to_token(answer_end)

    # Update the tokenized_input dictionary with token IDs
    tokenized_input['answer_start_token'] = answer_start_token
    tokenized_input['answer_end_token'] = answer_end_token

    return tokenized_input

def tokenize_csv(input_file, output_file):
    # Open the CSV file and process each row
    with open(input_file, 'r', newline='') as csv_input, \
            open(output_file, 'w', newline='') as csv_output:
        reader = csv.reader(csv_input)
        writer = csv.writer(csv_output)
    
        header = next(reader)  # Read the header row
        header.extend(['tokenized_context', 'tokenized_question', 'tokenized_answers'])
        writer.writerow(header)

        for row in reader:
            context = row[0]
            question = row[1]
            answers = row[3]

            tokenized_context = ' '.join(tokenizer(context,truncation=True, padding=True))
            tokenized_question = ' '.join(tokenizer(question,truncation=True, padding=True))
            tokenized_answers = ' '.join(tokenizer(answers,truncation=True, padding=True))

            row.extend([tokenized_context, tokenized_question, tokenized_answers])
            writer.writerow(row)

tokenize_csv(input_file, output_file)



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