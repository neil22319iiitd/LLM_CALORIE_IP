import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
from bert_score import BERTScorer
import torch
import nltk
from tqdm import tqdm  # To track progress

# Download necessary NLTK resources
nltk.download('punkt')

# Load your tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('./T5_BASE_TRAINING_MODEL/t5_tokenizer')  # Replace with your tokenizer path
model = T5ForConditionalGeneration.from_pretrained('./T5_BASE_TRAINING_MODEL/t5_model')  # Replace with your model path
model.eval()  # Set the model to evaluation mode

# Check if a GPU is available and move the model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load your testing dataset (modify as necessary)
test_df = pd.read_csv('tokenized_recipes_test.txt', delimiter="\t")  # Adjust as needed for your file format

# Get the first 400 examples
test_examples = test_df.iloc[:400]

# Prepare inputs and outputs
true_outputs = test_examples['true_recipe'].tolist()  # Replace with your actual output column name
predictions = []

# Process in batches of 8
batch_size = 8

for i in tqdm(range(0, len(test_examples), batch_size)):
    batch_examples = test_examples.iloc[i:i+batch_size]
    print(f"\nProcessing batch {i//batch_size + 1} of {len(test_examples)//batch_size + 1}")

    batch_inputs = batch_examples['input'].tolist()  # Replace with your input column name
    batch_true_outputs = batch_examples['true_recipe'].tolist()  # Replace with your output column name

    for input_text in batch_inputs:
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=150)  # Adjust max_length if needed
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(generated_text)

    print(f"Batch {i//batch_size + 1} processed.\n")

# Calculate BLEU Score
def calculate_bleu_score(predictions, true_outputs):
    bleu_scores = []
    for pred, true in zip(predictions, true_outputs):
        pred_tokens = nltk.word_tokenize(pred)
        true_tokens = nltk.word_tokenize(true)
        score = sentence_bleu([true_tokens], pred_tokens)
        bleu_scores.append(score)
    return sum(bleu_scores) / len(bleu_scores)

bleu_score = calculate_bleu_score(predictions, true_outputs)
print(f"BLEU Score: {bleu_score}")

# Calculate BERT Score
def calculate_bert_score(predictions, true_outputs):
    scorer = BERTScorer(lang='en', rescale_with_baseline=True)
    P, R, F1 = scorer.score(predictions, true_outputs)
    return F1.mean().item()  # Return average F1 score

bert_score = calculate_bert_score(predictions, true_outputs)
print(f"BERT Score: {bert_score}")

# Calculate Meteor Score
def calculate_meteor_score(predictions, true_outputs):
    meteor_scores = []
    for pred, true in zip(predictions, true_outputs):
        score = nltk.translate.meteor_score.meteor_score([true], pred)
        meteor_scores.append(score)
    return sum(meteor_scores) / len(meteor_scores)

meteor_score = calculate_meteor_score(predictions, true_outputs)
print(f"Meteor Score: {meteor_score}")
