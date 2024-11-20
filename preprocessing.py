import pandas as pd

# Function to preprocess the 'steps' field
def preprocess_steps(steps):
    # Split the steps based on full stops, removing extra spaces before/after
    step_list = [step.strip() for step in steps.split('.') if step.strip()]
    return step_list

# Load the CSV file
df = pd.read_csv('final_combined.csv')

# Apply preprocessing steps: lowercase all fields and split steps
df['steps'] = df['steps'].apply(lambda x: preprocess_steps(x.lower()))
df['ingredient_Phrase'] = df['ingredient_Phrase'].apply(lambda x: x.lower())
df['ingredient'] = df['ingredient'].apply(lambda x: x.lower())
df['Region'] = df['Region'].apply(lambda x: x.lower())
df['Sub_region'] = df['Sub_region'].apply(lambda x: x.lower())

# Save the updated dataframe to a new CSV file
df.to_csv('preprocessed_output.csv', index=False)

print("Preprocessing complete. File saved as 'preprocessed_output.csv'.")
