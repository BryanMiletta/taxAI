import create_dataset
from loadModel import *
import textwrap
p = create_dataset.Create_DS()
p.loadTxt("John is a 10 year old boy. He is the son of Robert Smith.  Elizabeth Davis is Robert's wife. She teaches at UC Berkeley. Sophia Smith is Elizabeth's daughter. She studies at UC Davis")
model = QAPipe(p.ds)
question = "Which college does John’s sister attend?"
answer_start_index,answer_end_index,start_token_score,end_token_score,s_Scores,e_Scores,answer=model.get_output(question)
wrapper = textwrap.TextWrapper(width=80)
print(wrapper.fill(p.ds)+"\n")

print("Question: "+question)
print("Answer : " + answer)

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

