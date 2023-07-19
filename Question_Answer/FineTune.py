#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: Fine-tuning

### ### ### Import necessary Libraries
import json
from pathlib import Path
from transformers import BertForQuestionAnswering, BertTokenizer
bert_model = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(bert_model)

### Accessor method for SQuAD json file
def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers
###

### Adds end index for answers and context
def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text) 
        
        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters
        else:    
            answer['answer_end'] = end_idx #TODO this might be causing an error.
        
###

### Adds token positions for answers
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for answer in answers:
        # Tokenize the answer text and get the tokenized answer
        answer_tokens = tokenizer.encode(answer['text'])
        
        # Find the start and end token positions in the tokenized input
        start_position = None
        end_position = None
        for i, token in enumerate(encodings.tokens[1:-1]):  # Exclude [CLS] and [SEP] tokens
            if token == answer_tokens[0]:
                if encodings.tokens[i+1:i+len(answer_tokens)-1] == answer_tokens[1:-1]:
                    start_position = i
                    end_position = i + len(answer_tokens) - 3
                    break

        start_positions.append(start_position)
        end_positions.append(end_position)

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
###