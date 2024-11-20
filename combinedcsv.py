import pandas as pd

# Load the ingredient_phrase CSV file
ingredient_phrase_df = pd.read_csv("RecipeDB_ingredient_phrase_fivecui.csv")

# Group by Recipe_id and aggregate the ingredient_Phrase, ingredient, and ing_id columns into lists
file1 = ingredient_phrase_df.groupby('Recipe_id').agg({
    'ingredient_Phrase': lambda x: list(x),
    'ingredient': lambda x: list(x),
    'ing_id': lambda x: list(x)
}).reset_index()

# Load the fivecui CSV file with low_memory set to False to avoid the warning
fivecui_df = pd.read_csv("RecipeDB_fivecui.csv", low_memory=False)

# Focus on the specified columns and merge with file1 based on Recipe_id
file2 = pd.merge(file1, fivecui_df[['Recipe_id', 'Calories', 'Region', 'Sub_region']], on='Recipe_id', how='left')

# Load the 5cui_instructions CSV file
instructions_df = pd.read_csv("RecipeDB_5cui_instructions.csv")

# Focus on the specified columns and merge with file2 based on Recipe_id
final_file = pd.merge(file2, instructions_df[['Recipe_id', 'steps']], on='Recipe_id', how='left')

# Save the final combined CSV file
final_file.to_csv("final_combined_file.csv", index=False)

print("The final CSV file has been saved as 'final_combined_file.csv'.")
