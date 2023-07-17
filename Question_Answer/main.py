
#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: Main run file to execute the model. UI to collect question, build dataset from 1040, run the data through the pre-training model. TODO run through fine-tuning model. Output result.

### ### ### Import necessary Libraries
from transformers import BertForQuestionAnswering, BertTokenizer

model_path = "db"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

import create_dataset
from loadModel import *
from run_fineTuning import *
import textwrap
import nltk
import PyPDF2
from PyPDF2 import PdfReader

### ### ### creates a dataset that pulls text from PDF - 
p = create_dataset.Create_DS()
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
p.loadTxt(extracted_text)
### ### ###

context = p.loadTxt(extracted_text)

question = input("\nPlease enter your question: \n")
while True:
    #model = QAPipe(p.ds)
    inputs = tokenizer(question, context, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens)

    wrapper = textwrap.TextWrapper(width=80)
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