
# Combining the Files

The first python script processes and combines multiple CSV files from the **RecipeDB** database to create a unified dataset. It merges information from ingredient details, nutritional data, regional classifications, and cooking instructions into a single file named `final_combined_file.csv`.

The script groups ingredient-related fields, merges them with metadata like calories and region, and finally appends cooking instructions. The result is a consolidated dataset, ready for further analysis


# Preprocessing the Data

This script preprocesses the `final_combined.csv` file by standardizing and cleaning its fields. It performs the following steps:  
1. Converts all fields (`steps`, `ingredient_Phrase`, `ingredient`, `Region`, `Sub_region`) to lowercase for uniformity.  
2. Splits the `steps` field into a list of individual instructions, removing unnecessary spaces and ensuring clear segmentation.
3. The processed data is saved as `preprocessed_output.csv`.

