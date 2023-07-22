#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: run file to execute the pre-training model. UI to collect question, build dataset from 1040, run the data through the pre-training model. TODO run through fine-tuning model. Output result.

### ### ### Import necessary Libraries
import create_dataset
import loadModel 
import textwrap
import nltk
import transformers
from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch
import numpy as np

'''
### ### ### creates a dataset that pulls text from PDF - 
import newspaper_sandbox
from newspaper import Article
import nltk
import PyPDF2
from PyPDF2 import PdfReader
p = newspaper_sandbox.Create_DS()
#url = 'https://en.wikipedia.org/wiki/Taxation_in_the_United_States'
#p.loadArticle(url)
# Step 1: Extract text from the PDF file
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
# Step 2: Use Newspaper3k to process the extracted text
def process_text(text):
    article = Article(text)
    article.set_text(text)
    article.parse()
    return article.title, article.text
# Step 3: Call the functions to extract and process the PDF text
pdf_file_path = 'db/f1040.pdf'
extracted_text = extract_text_from_pdf(pdf_file_path)
### ### ###
'''

#context = extracted_text
context = input("\nPlease enter your context: \n")

### ### ### PRE-TRAINING & Fine-Tuning
#model = QAPipe(context, model_path, tokenizer_path)
model_path = "db/model2"   
tokenizer_path = "db/model2/tokenizer2" 
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

### ### ###
# Function to get output
def get_output(question, context, model, tokenizer):
    max_length = 512
    input_ids = tokenizer.encode(question, context, max_length=max_length, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    SEP_index = input_ids.index(102)  # Getting index of first SEP token
    len_question = SEP_index + 1
    len_answer = len(input_ids) - len_question
    segment_ids = [0] * len_question + [1] * len_answer  # Segment Ids will 0 for question and 1 for answer

    # Getting start and end scores for the answer and converting input arrays to tensors
    start_token_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))[0]
    end_token_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))[1]

    # Converting scores tensor to numpy arrays so that we can use numpy functions
    start_token_scores = start_token_scores.detach().numpy().flatten()
    end_token_scores = end_token_scores.detach().numpy().flatten()

    # Picking start and end index of an answer based on start/end indices with highest scores
    answer_start_index = np.argmax(start_token_scores)
    answer_end_index = np.argmax(end_token_scores)

    # Getting Scores for start and end token of the answer.
    start_token_score = np.round(start_token_scores[answer_start_index], 2)
    end_token_score = np.round(end_token_scores[answer_end_index], 2)

    # Cleaning answer text
    answer = tokens[answer_start_index]
    for i in range(answer_start_index + 1, answer_end_index + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]

    return (
        answer_start_index,
        answer_end_index,
        start_token_score,
        end_token_score,
        start_token_scores,
        end_token_scores,
        answer,
    )

question = input("\nPlease enter your question: \n")
while True:
    #model = QAPipe(p.ds)
    #answer_start_index,answer_end_index,start_token_score,end_token_score,s_Scores,e_Scores,answer=model.get_output(question)
    # Get the output
    answer_start_index, answer_end_index, start_token_score, end_token_score, _, _, answer = get_output(question, context, model, tokenizer)

    wrapper = textwrap.TextWrapper(width=50)
    print() # space
    print("Question: " + question)
    print("Answer : " + answer)
    ###
    flag = True
    flag_N = False
    while flag:
        response = input("\nDo you want to ask another question based on this text (Y/N)? ")
        if response[0] == "Y":
            question = input("\nPlease enter your question: \n")
            flag = False
        elif response[0] == "N":
            print("\nBye!")
            flag = False
            flag_N = True      
    if flag_N == True:
        break
    ###

### ### ### Model details - outputs analytics on the model
'''
tokens = model.generate_text_from_token()
print("Passage: ")
print(wrapper.fill(context)+"\n")
print("Question: \n"+question+"\n")


print("Tokens: \n",tokens)
print("\nSegment Ids: \n",model.segment_ids)
print("\n Input Ids: \n" ,model.input_ids)

for token,id in zip(tokens,model.input_ids):
    if id == model.tokenizer.cls_token_id:
        print('')
    if id == model.tokenizer.sep_token_id:
        print('')
    print('{:<12} {:>6,}'.format(token,id))
    if id == model.tokenizer.cls_token_id:
        print('')
    if id== model.tokenizer.sep_token_id:
        print('')


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
plt.rcParams["figure.figsize"] = (16,8)

token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))

print(token_labels)

print(start_token_score)
print(answer_start_index)

# Start Word Scores
ax = sns.barplot(x=token_labels,y=s_Scores,ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
ax.grid(True)
plt.title("Start Word Scores")
plt.show()
# End Word Scores
ax =sns.barplot(x=token_labels,y=e_Scores,ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
ax.grid(True)
plt.title("End Word Scores")
plt.show()

# Visualizing in a single bar plot
import pandas as pd
#to store the tokens and scores in a Panda Dataframe
scores = []
for (i, token_label) in enumerate(token_labels):

    # Add the token's start score as one row.
    scores.append({'token_label': token_label, 
                   'score': s_Scores[i],
                   'marker': 'start'})
    
    # Add  the token's end score as another row.
    scores.append({'token_label': token_label, 
                   'score': e_Scores[i],
                   'marker': 'end'})
    
df = pd.DataFrame(scores)
# Draw a grouped barplot to show start and end scores for each word.
# The "hue" parameter is where we tell it which datapoints belong to which
# of the two series.
g = sns.catplot(x="token_label", y="score", hue="marker", data=df,
                kind="bar", height=6, aspect=4)

# Turn the xlabels vertical.
g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align words to scores.
g.ax.grid(True)

'''