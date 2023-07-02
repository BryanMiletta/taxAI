# creates the dataset
import newspaper_sandbox
from newspaper import Article
import nltk
#nltk.download('punkt')


p = newspaper_sandbox.Create_DS()
#p.loadTxt("John is a 10 year old boy. He is the son of Robert Smith.  Elizabeth Davis is Robert's wife. She teaches at UC Berkeley. Sophia Smith is Elizabeth's daughter. She studies at UC Davis")
#print(p.ds)

url = 'https://en.wikipedia.org/wiki/Taxation_in_the_United_States'
p.loadArticle(url)


