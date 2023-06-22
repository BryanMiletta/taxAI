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
# download the config file for pre-trained BERT model
import json
bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
config_dict

encoder_config = tfm.nlp.encoders.EncoderConfig({
    'type':'bert',
    'bert': config_dict
})

bert_encoder = tfm.nlp.encoders.build_encoder(encoder_config)
bert_encoder

bert_classifier = tfm.nlp.models.BertClassifier(network=bert_encoder, num_classes=2)

# run it on a test batch of data 10 examples from the training set. ouput is the logits for the two classes
bert_classifier(
    example_inputs, training=True).numpy()[:10]

tf.keras.utils.plot_model(bert_encoder, show_shapes=True, dpi=48)

### Restore the encoder weights
checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
checkpoint.read(
    os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()

### Set up the optimizer
# Set up epochs and steps
epochs = 5
batch_size = 32
eval_batch_size = 32

train_data_size = info.splits['train'].num_examples
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(0.1 * num_train_steps)
initial_learning_rate=2e-5

# linear decay from initial_learning_rate to zero over num_train_steps
linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    end_learning_rate=0,
    decay_steps=num_train_steps)
#warmup to that value over warmup_steps
warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
    warmup_learning_rate = 0,
    after_warmup_lr_sched = linear_decay,
    warmup_steps = warmup_steps
)
# overall schedule looks like:
x = tf.linspace(0, num_train_steps, 1001)
y = [warmup_schedule(xi) for xi in x]
plt.plot(x,y)
plt.xlabel('Train step')
plt.ylabel('Learning rate')

# Use tf.keras.optimizers.experimental.AdamW to instantiate the optimizer with that schedule:
optimizer = tf.keras.optimizers.experimental.Adam(
    learning_rate = warmup_schedule)

### Train the model
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)
bert_classifier.evaluate(glue_validation)
bert_classifier.fit(
      glue_train,
      validation_data=(glue_validation),
      batch_size=32,
      epochs=epochs)

### ### Now run the fine-tuned model on a custom example to see that it works:
# start by encoding some sentence pairs
my_examples = {
        'sentence1':[
            'The rain in Spain falls mainly on the plain.',
            'Look I fine tuned BERT.'],
        'sentence2':[
            'It mostly rains on the flat lands of Spain.',
            'Is it working? This does not match.']
    }
# the model should report class 1 "match" for the first example and class 0 "no-match" for the second
ex_packed = bert_inputs_processor(my_examples)
my_logits = bert_classifier(ex_packed, training=False)

result_cls_ids = tf.argmax(my_logits)
result_cls_ids
tf.gather(tf.constant(info.features['label'].names), result_cls_ids)

### Export the model

