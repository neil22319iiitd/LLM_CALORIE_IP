import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from tqdm import tqdm

# Load the dataset
file_path = 'preprocessed_output.csv'
data = pd.read_csv(file_path)

# Filter to the first 500 lines and get the ingredients field
ingredients_list = data['ingredient'].head(500)

# Load the GPT-2 model and tokenizer
model_path = 'GPT2_NEW'  # Adjust to your GPT-2 model's path if necessary
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate recipe based on a list of ingredients
def generate_recipe(ingredients):
    input_text = ', '.join(ingredients) + ";"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=200,   # Adjust as needed for recipe length
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Generate recipes and save to a file
output_file_path = 'generated_recipes.txt'
with open(output_file_path, 'w') as f:
    for ingredients in tqdm(ingredients_list, desc="Generating recipes"):
        ingredients = eval(ingredients)  # Convert string representation of list to actual list
        recipe = generate_recipe(ingredients)
        f.write(recipe + '\n')

print(f"Recipes have been saved to {output_file_path}")
