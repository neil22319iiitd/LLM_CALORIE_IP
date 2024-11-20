
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

# RecipeDB T5 Model Training 

The fourth script sets up a training pipeline for a T5-based model to generate recipes using a structured, tokenized dataset. The model processes recipes formatted with custom tokens to segment titles, ingredients, and instructions for structured input and output. The custom tokens used include `<BEGIN_RECIPE>`, `<END_RECIPE>`, `<BEGIN_INPUT>`, `<NEXT_INPUT>`, `<END_INPUT>`, `<BEGIN_TITLE>`, `<END_TITLE>`, `<BEGIN_INGREDS>`, `<NEXT_INGREDS>`, `<END_INGREDS>`, `<BEGIN_INSTR>`, `<NEXT_INSTR>`, and `<END_INSTR>`. The script starts by loading and splitting the dataset into training and testing sets. It initializes a tokenizer, adds the custom tokens, and prepares a T5 model by resizing its token embeddings to include the newly added tokens. Data loaders for training and testing are created with a reduced batch size for efficient computation, especially on limited hardware. Mixed precision (fp16) training is enabled for faster processing. The model undergoes training for three epochs with evaluation steps, and the best version is saved for further use. 

### Note: This T5 model has not been used finally. It was only tested as an alternative to the GPT-2 model used in the *Ratatouille* project for recipe generation.


# QUALITY SCORES 

ThE fifth script evaluates the performance of a trained T5-based recipe generation model using three metrics: **BLEU**, **BERTScore**, and **METEOR**. The evaluation is conducted on a testing dataset containing tokenized inputs and true recipe outputs. The script processes inputs in batches of 8 for efficient computation and generates predictions using the trained model. It then compares these predictions against the true outputs to calculate the evaluation scores.

The script begins by loading a T5 tokenizer and model, both of which have been trained and saved locally. The model is set to evaluation mode and moved to a GPU if available. It loads a testing dataset, extracts the first 400 examples for evaluation, and processes them in batches. For each input, the script generates predictions using the model and decodes them into human-readable text.

The evaluation results achieved with this script were as follows:
- **BLEU Score:** 0.94  
- **BERTScore:** 0.45  
- **METEOR Score:** 0.37  

The **BLEU score** measures the overlap of n-grams between the generated and true outputs, achieving a high score of 0.94, indicating significant lexical similarity. The **BERTScore**, which leverages semantic similarity, scored 0.45, reflecting moderate semantic alignment. Lastly, the **METEOR score**, which considers synonymy and alignment, was 0.37, indicating room for improvement in capturing linguistic nuances.

This evaluation script provides a robust framework for assessing the quality and accuracy of recipe generation by examining both lexical and semantic alignments between generated and ground-truth outputs.
