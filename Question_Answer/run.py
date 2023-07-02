# Import the model from loadModel.py
import create_dataset
from loadModel import *
import textwrap
import nltk

### ### ### creates a dataset that pulls text from website
#p = create_dataset.Create_DS()
#url = 'https://en.wikipedia.org/wiki/Taxation_in_the_United_States'
#p.loadArticle(url)
### ### ###

### ### ### creates a dataset that pulls text from PDF
#p = create_dataset.Create_DS()
#url = 'https://en.wikipedia.org/wiki/Taxation_in_the_United_States'
#p.loadArticle(url)
### ### ###

# creates the dataset that pulls data from hardcoded text
#p = create_dataset.Create_DS()
#p.loadTxt('State rules for determining taxable income often differ from federal rules. Federal marginal tax rates vary from 10% to 37% of taxable income.[3] State and local tax rates vary widely by jurisdiction, from 0% to 13.30% of income,[4] and many are graduated. State taxes are generally treated as a deductible expense for federal tax computation, although the 2017 tax law imposed a $10,000 limit on the state and local tax ("SALT") deduction, which raised the effective tax rate on medium and high earners in high tax states. Prior to the SALT deduction limit, the average deduction exceeded $10,000 in most of the Midwest, and exceeded $11,000 in most of the Northeastern United States, as well as California and Oregon.[5] The states impacted the most by the limit were the tri-state area (NY, NJ, and CT) and California; the average SALT deduction in those states was greater than $17,000 in 2014.[5]')
### ### ###

model = QAPipe(p.ds)
question = "What is the range of state and local tax rates?"

answer_start_index,answer_end_index,start_token_score,end_token_score,s_Scores,e_Scores,answer=model.get_output(question)
wrapper = textwrap.TextWrapper(width=80)
print(wrapper.fill(p.ds)+"\n")

print("Question: "+question)
print("Answer : " + answer)

### ### ### Model details
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

