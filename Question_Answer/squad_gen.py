#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: Generates the SQuAD json traing file

### ### ### Import necessary Libraries
import create_dataset

### ### ### creates a dataset using text from:
# https://www.irs.gov/taxtopics 
# https://www.irs.gov/faqs 

p = create_dataset.Create_DS()
url = 'https://en.wikipedia.org/wiki/Taxation_in_the_United_States'
input_text = p.loadArticle(url)

### ### ###
# creates the dataset that pulls data from hardcoded text
#p = create_dataset.Create_DS()
#p.loadTxt('Text')
### ### ###

# Read your text file
with open(input_text, 'r', encoding='utf-8') as f:
    text_data = f.read()

# Convert text data to SQuAD format
squad_data = create_squad(text_data)

# Save the SQuAD data to a JSON file
with open('db', 'w', encoding='utf-8') as f:
    f.write(squad_data)