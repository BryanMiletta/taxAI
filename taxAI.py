#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Sandbox
#summary:this code is testing setting up a Q/A BERT model that allows user input. 
#it accesses local folders that already have text files and uses them for training. Specifically this uses pos neg movie reviews. It trains, tests, and evaluates


### ### ### Import Libraries
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering, TFBertForQuestionAnswering, TFBertModel
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV2Processor
from transformers.data.metrics.squad_metrics import squad_evaluate

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def answer_question(context, question):
    # Preprocess the context and question
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='tf')

    # Retrieve the input IDs and attention masks
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Perform the question answering inference
    start_scores, end_scores = model(input_ids, attention_mask=attention_mask)

    # Get the most probable start and end positions
    start_index = tf.argmax(start_scores, axis=1).numpy()[0]
    end_index = tf.argmax(end_scores, axis=1).numpy()[0]

    # Convert the token IDs to actual tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.numpy()[0])
    answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])

    return answer

# Load the SQuAD dataset
processor = SquadV2Processor()
train_examples = processor.get_train_examples('path/to/train.json')
dev_examples = processor.get_dev_examples('path/to/dev.json')

# Convert examples to features
train_features = squad_convert_examples_to_features(
    examples=train_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=True,
    return_dataset="tf",
    threads=1,
)

dev_features = squad_convert_examples_to_features(
    examples=dev_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    return_dataset="tf",
    threads=1,
)

# Create a TensorFlow dataset from the features
train_dataset = train_features["dataset"]
dev_dataset = dev_features["dataset"]

# Fine-tune the BERT model on the SQuAD dataset
model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss)

model.fit(train_dataset.shuffle(100).batch(16), epochs=2, batch_size=16)

# Evaluate the fine-tuned model
results = model.evaluate(dev_dataset.batch(16))
print("Evaluation results:", results)

# Example usage
context = "The quick brown fox jumps over the lazy dog."
question = "What does the fox jump over?"
answer = answer_question(context, question)
print(answer)