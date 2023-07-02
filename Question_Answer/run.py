# Import the model from loadModel.py
import create_dataset
from loadModel import *
import textwrap
import nltk
import PyPDF2
from PyPDF2 import PdfReader

question = input("\nPlease enter your question: \n")

### ### ### creates a dataset that pulls text from website
#p = create_dataset.Create_DS()
#url = 'https://en.wikipedia.org/wiki/Taxation_in_the_United_States'
#p.loadArticle(url)
### ### ###

### ### ### creates a dataset that pulls text from PDF
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

# creates the dataset that pulls data from hardcoded text
#p = create_dataset.Create_DS()
#p.loadTxt('The United States of America has separate federal, state, and local governments with taxes imposed at each of these levels. Taxes are levied on income, payroll, property, sales, capital gains, dividends, imports, estates and gifts, as well as various fees. In 2020, taxes collected by federal, state, and local governments amounted to 25.5% of GDP, below the OECD average of 33.5% of GDP. The United States had the seventh-lowest tax revenue-to-GDP ratio among OECD countries in 2020, with a higher ratio than Mexico, Colombia, Chile, Ireland, Costa Rica, and Turkey.[1] U.S. tax and transfer policies are progressive and therefore reduce effective income inequality, as rates of tax generally increase as taxable income increases. As a group, the lowest earning workers, especially those with dependents, pay no income taxes and may actually receive a small subsidy from the federal government (from child credits and the Earned Income Tax Credit).[2] Taxes fall much more heavily on labor income than on capital income. Divergent taxes and subsidies for different forms of income and spending can also constitute a form of indirect taxation of some activities over others. Taxes are imposed on net income of individuals and corporations by the federal, most state, and some local governments. Citizens and residents are taxed on worldwide income and allowed a credit for foreign taxes. Income subject to tax is determined under tax accounting rules, not financial accounting principles, and includes almost all income from whatever source. Most business expenses reduce taxable income, though limits apply to a few expenses. Individuals are permitted to reduce taxable income by personal allowances and certain non-business expenses, including home mortgage interest, state and local taxes, charitable contributions, and medical and certain other expenses incurred above certain percentages of income.')
### ### ###

while True:
    model = QAPipe(p.ds)
    answer_start_index,answer_end_index,start_token_score,end_token_score,s_Scores,e_Scores,answer=model.get_output(question)
    wrapper = textwrap.TextWrapper(width=80)
    #print(wrapper.fill(p.ds)+"\n") # this prints the context, not needed for the user
    print() # space
    print("Question: "+question)
    print("Answer : " + answer)

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

### ### ### Model details
'''
tokens = model.generate_text_from_token()
print("Passage: ")
print(wrapper.fill(p.ds)+"\n")
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