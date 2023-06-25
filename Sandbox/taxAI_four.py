#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Sandbox
#summary:this code uses tensorflow with BERT

### ### ### Import Libraries
import tensorflow as tf
from transformers import TFBertForQuestionAnswering, BertTokenizer
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV2Processor
from transformers.data.metrics.squad_metrics import compute_predictions_logits

# Load the fine-tuned BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = TFBertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define the SQuAD processor for data conversion
squad_processor = SquadV2Processor()

# Load the SQuAD training dataset
train_dataset = squad_processor.get_train_examples('path_to_train_dataset')
train_features, train_dataset = squad_convert_examples_to_features(
    examples=train_dataset,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=True,
    return_dataset="tf"
)
train_dataloader = tf.data.Dataset.from_tensor_slices(train_features).shuffle(100).batch(16)

# Fine-tune the BERT model with SQuAD training dataset
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_positions = inputs['start_positions']
        end_positions = inputs['end_positions']
        start_loss = loss_fn(start_positions, start_logits)
        end_loss = loss_fn(end_positions, end_logits)
        total_loss = start_loss + end_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

for epoch in range(3):  # Example: Run 3 epochs
    epoch_loss = 0.0
    for step, inputs in enumerate(train_dataloader):
        loss = train_step(inputs)
        epoch_loss += loss
        if (step + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {epoch_loss / (step + 1):.4f}")

# Load the SQuAD evaluation dataset
eval_dataset = squad_processor.get_dev_examples('path_to_eval_dataset')
eval_features, eval_dataset = squad_convert_examples_to_features(
    examples=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    return_dataset="tf"
)
eval_dataloader = tf.data.Dataset.from_tensor_slices(eval_features).batch(16)

# Evaluate the fine-tuned BERT model on SQuAD evaluation dataset
all_results = []
for inputs in eval_dataloader:
    outputs = model(inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    for i in range(len(inputs['input_ids'])):
        result = SquadResult(unique_id=inputs['unique_ids'][i].numpy(), start_logits=start_logits[i].numpy(), end_logits=end_logits[i].numpy())
        all_results.append(result)

# Convert model output to predictions
eval_examples = eval_dataset.map(lambda x, y: x)
eval_features = tf.data.Dataset.from_tensor_slices(eval_features)
prediction_dataset = tf.data.Dataset.zip((eval_features, tf.data.Dataset.from_tensor_slices(all_results)))
preds = prediction_dataset.batch(16).map(lambda x, y: squad_processor.add_predictions_to_examples(x, y)).flat_map(lambda x: x)

# Compute final predictions
all_predictions = compute_predictions_logits(
    eval_examples,
    preds,
    n_best_size=20,
    max_answer_length=30,
    output_prediction_file="predictions.json",
    output_nbest_file="nbest_predictions.json",
    output_null_log_odds_file="null_odds.json",
    verbose_logging=False,
    version_2_with_negative=True,
    null_score_diff_threshold=0.0,
)

# Print predictions
for key in all_predictions.keys():
    print('Question:', key)
    print('Answer:', all_predictions[key])

# Save the fine-tuned BERT model
model.save_pretrained('path_to_save_model')
tokenizer.save_pretrained('path_to_save_model')
