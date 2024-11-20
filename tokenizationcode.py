import pandas as pd
from transformers import T5Tokenizer

# Define custom tokens
custom_tokens = [
    "<BEGIN_RECIPE>", "<END_RECIPE>", 
    "<BEGIN_INPUT>", "<NEXT_INPUT>", "<END_INPUT>",
    "<BEGIN_TITLE>", "<END_TITLE>",
    "<BEGIN_INGREDS>", "<NEXT_INGREDS>", "<END_INGREDS>",
    "<BEGIN_INSTR>", "<NEXT_INSTR>", "<END_INSTR>"
]

# Function to convert DataFrame to a formatted plain text file using custom tokens
def df_to_plaintext_file(input_df, output_file):
    print("Writing to", output_file)
    with open(output_file, 'w', encoding="utf-8") as f:
        for index, row in input_df.iterrows():
            title = row['Recipe_id']  # Assuming 'Recipe_id' is the title (change if needed)
            instructions = eval(row['steps'])  # Steps are tokenized as list already
            ingredients = row['ingredient_Phrase'].strip('[]').split(', ')  # List of ingredients
            keyword = row['ingredient'].strip('[]').split(', ')  # Assuming these are the base ingredients

            # Log progress for every 40,000 recipes
            if index % 40000 == 0:
                print(index)
                print("Ingredients --->", ingredients)
                print("Keywords --->", keyword)

            # Construct the tokenized recipe format
            res = "<BEGIN_RECIPE> <BEGIN_INPUT> " + " <NEXT_INPUT> ".join(keyword) + " <END_INPUT> <BEGIN_TITLE> " + \
                  title + " <END_TITLE> <BEGIN_INGREDS> " + \
                  " <NEXT_INGREDS> ".join(ingredients) + " <END_INGREDS> <BEGIN_INSTR> " + \
                  " <NEXT_INSTR> ".join(instructions) + " <END_INSTR> <END_RECIPE>"

            # Write the formatted recipe to the file
            f.write("{}\n".format(res))

# Load the preprocessed CSV file
df = pd.read_csv('preprocessed_output.csv')

# Write the formatted file
df_to_plaintext_file(df, 'formatted_recipes.txt')

# Initialize the T5 tokenizer (using 't5-base' for example)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Add custom tokens to the tokenizer
tokenizer.add_tokens(custom_tokens)

# Save the tokenizer with the added tokens
tokenizer.save_pretrained('./tokenizer_with_custom_tokens')

# Now, tokenize the text using the tokenizer with the added special tokens
def tokenize_recipes(input_file, output_file, tokenizer):
    with open(input_file, 'r', encoding="utf-8") as infile, open(output_file, 'w', encoding="utf-8") as outfile:
        for line in infile:
            # Tokenize the line with special tokens
            tokenized_output = tokenizer.encode(line, return_tensors="pt", add_special_tokens=True)
            outfile.write(f"{tokenized_output.tolist()}\n")

# Tokenize the formatted recipes and save the output
tokenize_recipes('formatted_recipes.txt', 'tokenized_recipes.txt', tokenizer)

print("Tokenization complete. Tokenized recipes saved in 'tokenized_recipes.txt'.")
