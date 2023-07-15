import pandas as pd
import json

def convert_csv_to_squad(csv_file, json_file):
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Initialize SQuAD JSON structure
    squad_data = {
        'data': []
    }

    paragraphs = []

    # Iterate over each row in the CSV
    for index, row in df.iterrows():
        context = row['context']
        question = row['question']
        answers = row['answers']
        answer_start = row['answer_start']

        # Create a SQuAD paragraph entry
        paragraph = {
            'context': context,
            'qas': [{
                'question': question,
                'id': str(index),
                'answers': [{
                    'text': answers,
                    'answer_start': answer_start
                }]
            }]
        }

        paragraphs.append(paragraph)

    # Create a SQuAD data entry
    squad_data['data'].append({
        'title': 'TaxAI Dataset',
        'paragraphs': paragraphs
    })

    # Save the SQuAD JSON file
    with open(json_file, 'w') as outfile:
        json.dump(squad_data, outfile)

# Convert CSV to SQuAD JSON
csv_file = 'db/updated_csv.csv'
json_file = 'db/json_file.json'
convert_csv_to_squad(csv_file, json_file)

