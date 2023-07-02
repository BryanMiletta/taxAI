# Import the model from loadModel.py
import create_dataset
from loadModel import *
import textwrap

# creates the dataset
p = create_dataset.Create_DS()
p.loadTxt("John is a 10 year old boy. He is the son of Robert Smith.  Elizabeth Davis is Robert's wife. She teaches at UC Berkeley. Sophia Smith is Elizabeth's daughter. She studies at UC Davis")
model = QAPipe(p.ds)
question = "Which college does John’s sister attend?"

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

