#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Sandbox
#summary:this code is testing setting up a Q/A BERT model that allows user input. 
#it accesses local folders that already have text files and uses them for training. Specifically this uses pos neg movie reviews. It trains, tests, and evaluates


### ### ### Import Libraries
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from transformers import AutoTokenizer, TFAutoModel
### ### ###

### ### ### Data processing
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'db/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)
class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = tf.keras.utils.text_dataset_from_directory(
    'db/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = tf.keras.utils.text_dataset_from_directory(
    'db/test',
    batch_size=batch_size)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
### ### ### END Data processing 

### ### ### Functions to read data
def readTextExamples(folder,cl,n) :
    """ Reads maximum n text files from folder and returns them as a list of
        list of text and its class label cl."""
    from glob import glob
    x = [] # list of text examples and class labels

    files = glob(folder+"/*.txt") # read all text files files
    for file in files :
        infile = open(file,"r",encoding="utf-8")
        data = infile.read()
        infile.close()
        x.append([data,cl])
        if len(x)==n :
            break
    return x

def readPosNeg(pos_folder,neg_folder,n) :
    """ Reads maximum n positive and maximum n negative text examples
        from the respective folders. Randomizes them. Returns the list of
        text and an np.array of corresponding one-hot class labels [1,0] for pos
        and [0,1] for neg. """
    pos = readTextExamples(pos_folder,[1,0],n)
    neg = readTextExamples(neg_folder,[0,1],n)
    allEg = pos + neg
    random.shuffle(allEg)
    x = []
    y = []
    for eg in allEg :
        x.append(eg[0])
        y.append(eg[1])
    return x, np.array(y)
### ### ### END Functions to read data

### ### ### train and test 
def train_test() :
    """ Training and testing for text classification using BERT. """
    maxlen=512 # 512 maximum number of tokens
    maxTrEg=1000 # maximum number of pos & neg training examples
    maxTeEg=1000 # maximum number of pos & neg test examples
    epochs=3 # number of epochs

    # read the data
    train_x, train_y = readPosNeg("db/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("db/test/pos","aclImdb/test/neg",maxTeEg)

    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")

    # build the model
    bert_model = TFAutoModel.from_pretrained("bert-base-cased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)

    output = Dense(2,activation="softmax")(bert_output[0][:,0])

    model = Model(inputs=[token_ids,attention_masks],outputs=output)

    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=epochs)

    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)
    
    print("Accuracy on test data:",score[1])

    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]], test_y
### ### ###


# Run the code
train_test()
#End