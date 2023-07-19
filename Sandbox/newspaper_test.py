# creates the dataset
import newspaper_sandbox
from newspaper import Article
import nltk
import PyPDF2
from PyPDF2 import PdfReader

p = newspaper_sandbox.Create_DS()
#p.loadTxt("John is a 10 year old boy. He is the son of Robert Smith.  Elizabeth Davis is Robert's wife. She teaches at UC Berkeley. Sophia Smith is Elizabeth's daughter. She studies at UC Davis")
#print(p.ds)

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
pdf_file_path = 'db/f1040_filled.pdf'
extracted_text = extract_text_from_pdf(pdf_file_path)
print(extracted_text)
#title, text = process_text(extracted_text)

# Step 4: Print the extracted title and text
#print("Title:", title)
#print("Text:", text)
#print(p.ds) # used to print URL

        

