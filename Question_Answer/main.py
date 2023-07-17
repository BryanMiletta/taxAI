
#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: Main run file to execute the model. UI to collect question, build dataset from 1040, run the data through the pre-training model. TODO run through fine-tuning model. Output result.

### ### ### Import necessary Libraries
import run.py
import run_fineTuning.py

def main(_):
    p = create_dataset.Create_DS()
    
    model = QAPipe(p.ds)
    #result = model.get_output("What happens after life")
   # import IPython; IPython.embed(); exit(1);

if __name__ == '__main__':
    app.run(main)