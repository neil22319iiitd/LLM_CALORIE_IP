import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import random

# Load the tokenizer and model from the saved path
model_path = "./T5_BASE_TRAINING_MODEL/t5_model"
tokenizer_path = "./T5_BASE_TRAINING_MODEL/t5_tokenizer"

tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the preprocessed dataset
df = pd.read_csv('/mnt/data/preprocessed_output.csv')

# Select random ingredients and phrases for novel recipe generation
def select_random_ingredients(df):
    random_recipe = df.sample().iloc[0]
    ingredient_phrases = random_recipe['ingredient_Phrase'].strip('[]').split(', ')
    ingredients = random_recipe['ingredient'].strip('[]').split(', ')
    
    # Prepare the input format
    input_text = (
        f"<BEGIN_RECIPE> <BEGIN_INPUT> {' <NEXT_INPUT> '.join(ingredient_phrases)} "
        f"<END_INPUT> <BEGIN_INGREDS> {' <NEXT_INGREDS> '.join(ingredients)} <END_INGREDS>"
    )
    
    return input_text

# Function to generate novel recipes
def generate_recipe(model, tokenizer, input_text, max_length=200):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate the output using the model
    output = model.generate(input_ids, max_length=max_length, num_beams=5, early_stopping=True)
    
    # Decode the output tokens to text
    recipe = tokenizer.decode(output[0], skip_special_tokens=False)
    
    return recipe

# Function to generate and save multiple recipes
def generate_novel_recipes(model, tokenizer, df, output_file, num_recipes=5):
    with open(output_file, 'w', encoding='utf-8') as f:
        for _ in range(num_recipes):
            # Select random ingredients for the recipe
            input_text = select_random_ingredients(df)
            
            # Generate the recipe
            generated_recipe = generate_recipe(model, tokenizer, input_text)
            
            # Write the generated recipe to the output file
            f.write(f"{generated_recipe}\n\n")
            
            print(f"Generated recipe: {generated_recipe}")

# Generate 5 novel recipes and save them to a file
output_file = "NovelRecipesGenerated.txt"
generate_novel_recipes(model, tokenizer, df, output_file, num_recipes=5)

print(f"Generated recipes saved to {output_file}")
