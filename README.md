
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

# Novel Recipe Generation Script
This script generates novel recipes using a fine-tuned T5-based model. It randomly selects ingredients and ingredient phrases from a preprocessed dataset, formats them into a structured input, and then uses the T5 model to generate creative recipe instructions.

The script begins by loading the pre-trained T5 model and tokenizer from the specified file paths. It then samples random recipes from the provided dataset, extracting ingredient phrases and base ingredients. These are formatted into an input string using custom tokens: `<BEGIN_RECIPE>`, `<BEGIN_INPUT>`, `<NEXT_INPUT>`, `<END_INPUT>`, `<BEGIN_INGREDS>`, `<NEXT_INGREDS>`, and `<END_INGREDS>`. 

For each selected recipe, the input string is passed to the T5 model, which generates a corresponding recipe using beam search decoding. The generated recipe is decoded and saved to an output file. By default, the script generates five unique recipes, but this can be adjusted based on the user’s needs. Each recipe is stored in a separate line in the output file.

This script serves as a tool for generating diverse and novel recipes by leveraging a fine-tuned T5 model, showcasing the model's potential in recipe creation based on random ingredients and phrases from the dataset.


# Shifting to GPT-2
This script generates recipes using a fine-tuned GPT-2 model based on a list of ingredients. Initially, the script loads a preprocessed dataset of ingredients, selects the first 500 lines, and processes them to create recipe inputs for GPT-2. The ingredients are used to generate a structured recipe by feeding them into the GPT-2 model. The model then generates a recipe text that is saved to a file.

After experimenting with T5-based models and not achieving satisfactory results, we decided to shift to GPT-2, which is more suitable for this kind of generative task. The GPT-2 model is capable of producing coherent and diverse outputs, especially for creative text generation tasks like recipe creation.

The script works by first reading the preprocessed dataset and extracting the ingredients for the first 500 entries. The ingredients are formatted into a string and passed to the GPT-2 model, which then generates a recipe based on that input. The generated recipes are stored in an output file.

The parameters used for generating recipes in GPT-2 include:
- `max_length=200` to control the maximum length of the generated recipe
- `temperature=1.0` for randomness in generation
- `top_k=50` and `top_p=0.95` for controlling the diversity of the generated text using sampling
- `do_sample=True` to enable sampling, allowing the model to generate different outputs for the same input.

The generated recipes are saved to a text file, with one recipe per line. 

Note: This approach was chosen after extensive experimentation with T5-based models, where the results were not as satisfactory for generating diverse recipes. The GPT-2 model has shown improved performance and creativity in this task.

