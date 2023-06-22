#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Sandbox
#summary:this code follows example from https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert 

### ### ### Import Libraries
import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

### ### ### Resources
gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)

### ### ### Load and Preprocess the dataset
# loading the MRPC dataset from TFDS
batch_size=32
glue, info = tfds.load('glue/mrpc',
                       with_info=True,
                       batch_size=32)
glue

# the info object describes the dataset and its features
info.features
# the two classes are
info.features['label'].names
# one example from the training set
example_batch = next(iter(glue['train']))

for key, value in example_batch.items():
  print(f"{key:9s}: {value[0].numpy()}")

### Preprocess the dataset
# the following code rebuilds the tokenizer that was used by the base model using Model Garden's layer
tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
    lower_case=True)
# tokenize the sentence
tokens = tokenizer(tf.constant(["Hello TensorFlow!"]))
tokens
# pack the inputs
special = tokenizer.get_special_tokens_dict()
special

max_seq_length = 128

packer = tfm.nlp.layers.BertPackInputs(
    seq_length=max_seq_length,
    special_tokens_dict = tokenizer.get_special_tokens_dict())
# example takes a list of tokenized sentences as input
sentences1 = ["hello tensorflow"]
tok1 = tokenizer(sentences1)
tok1
sentences2 = ["goodbye tensorflow"]
tok2 = tokenizer(sentences2)
tok2
# returns a dictionary containing three outputs: wordids, mask, typeids
packed = packer([tok1, tok2])

for key, tensor in packed.items():
  print(f"{key:15s}: {tensor[:, :12]}")

### combine these two parts into a keras.layers.Layer that can be attached to our model
class BertInputProcessor(tf.keras.layers.Layer):
  def __init__(self, tokenizer, packer):
    super().__init__()
    self.tokenizer = tokenizer
    self.packer = packer

  def call(self, inputs):
    tok1 = self.tokenizer(inputs['sentence1'])
    tok2 = self.tokenizer(inputs['sentence2'])

    packed = self.packer([tok1, tok2])

    if 'label' in inputs:
      return packed, inputs['label']
    else:
      return packed

# BUT for now just apply it to the dataset using Dataset.map since the dataset we loaded from TFDS is a tf.data.Dataset object
bert_inputs_processor = BertInputProcessor(tokenizer, packer)
glue_train = glue['train'].map(bert_inputs_processor).prefetch(1)

# example batch from the processed dataset
example_inputs, example_labels = next(iter(glue_train))
example_inputs
example_labels
for key, value in example_inputs.items():
  print(f'{key:15s} shape: {value.shape}')

print(f'{"labels":15s} shape: {example_labels.shape}')

### PLOTS
# the input_words_ids contain the token IDs:
plt.pcolormesh(example_inputs['input_word_ids'])

# the mask allows the model to cleanly differentiate between the content and the padding. The mask has the same shape as the input_word_ids, and contains a 1 anywhere the input_word_ids is not padding
plt.pcolormesh(example_inputs['input_mask'])

# The "input type" also has the same shape, but inside the non-padded region, contains a 0 or a 1 indicating which sentence the token is a part of.
plt.pcolormesh(example_inputs['input_type_ids'])

### Apply the same preprocessing to the validation and test subsets of the GLUE MRPC dataset:
glue_validation = glue['validation'].map(bert_inputs_processor).prefetch(1)
glue_test = glue['test'].map(bert_inputs_processor).prefetch(1)

### ### ### Build, train and export the model
