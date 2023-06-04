#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Sandbox
#summary:this code is testing setting up a Q/A BERT model that allows user input. It is not fine tuned yet

### ### ### Import Libraries
import torch
### ### ###

### ### ### Transformers libray download and selections 
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline 
### ### ###
### ### ### - model and tokenizer set up
# Choose a pre-trained BERT model
model_name = "bert-base-uncased"
nlp = pipeline("question-answering")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
### ### ###

### ### ### UI
# Initial input of context from the user
context = input("Enter your context: ")
question = "" #initilizes question

# while loop to continue to prompt question/answers until user enters "exit"
while question != "exit":
    # promt the user for a question
    question = input("Enter your question (type exit to terminate the session): ")
   
    # watch for request to exit the program
    if question == "exit": 
        print("OK we are going to exit this session.") 
        break   # terminates the program
    
    # Tokenize the input
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    print("***debug output: ",inputs)
    print()

    ### runs pipeline model
    answer_pipeline = nlp(question=question, context=context)
    ###
    
    # Perform the question-answering task
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the predicted answer span
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    # Output
    print("Question:", question)
    print("Answer for bert-base-uncased:", answer)
    print("Answer for pipeline:", answer_pipeline["answer"])
    print("Answer for pipeline:", answer_pipeline["score"])
    print()

# endWhile




