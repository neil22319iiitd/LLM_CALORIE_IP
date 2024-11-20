
# Combining the Files

The first python script processes and combines multiple CSV files from the **RecipeDB** database to create a unified dataset. It merges information from ingredient details, nutritional data, regional classifications, and cooking instructions into a single file named `final_combined_file.csv`.

The script groups ingredient-related fields, merges them with metadata like calories and region, and finally appends cooking instructions. The result is a consolidated dataset, ready for further analysis


# Preprocessing the Data

The second script preprocesses the `final_combined.csv` file by standardizing and cleaning its fields. It performs the following steps:  
1. Converts all fields (`steps`, `ingredient_Phrase`, `ingredient`, `Region`, `Sub_region`) to lowercase for uniformity.  
2. Splits the `steps` field into a list of individual instructions, removing unnecessary spaces and ensuring clear segmentation.
3. The processed data is saved as `preprocessed_output.csv`.


# Tokenization of the database

The third  script processes a preprocessed recipe dataset (`preprocessed_output.csv`) and converts it into a structured, tokenized format suitable for training language models like T5. It uses **custom tokens** to organize the data into logical sections and ensures consistency for downstream tasks.  

The custom tokens used in the script are as follows:  
- **Recipe Delimiters**: `<BEGIN_RECIPE>`, `<END_RECIPE>`  
- **Input Section**: `<BEGIN_INPUT>`, `<NEXT_INPUT>`, `<END_INPUT>`  
- **Title Section**: `<BEGIN_TITLE>`, `<END_TITLE>`  
- **Ingredients Section**: `<BEGIN_INGREDS>`, `<NEXT_INGREDS>`, `<END_INGREDS>`  
- **Instructions Section**: `<BEGIN_INSTR>`, `<NEXT_INSTR>`, `<END_INSTR>`  

The script first formats recipes into a plain text file (`formatted_recipes.txt`), where each recipe is clearly segmented into base ingredients, ingredient phrases, and cooking instructions using the custom tokens. The formatted file is then tokenized using an extended T5 tokenizer that recognizes the custom tokens. The tokenized recipes are saved in `tokenized_recipes.txt`, and the updated tokenizer is stored in the `./tokenizer_with_custom_tokens` directory.

