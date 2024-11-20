import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd

# Custom tokens as before
custom_tokens = [
    "<BEGIN_RECIPE>", "<END_RECIPE>",
    "<BEGIN_INPUT>", "<NEXT_INPUT>", "<END_INPUT>",
    "<BEGIN_TITLE>", "<END_TITLE>",
    "<BEGIN_INGREDS>", "<NEXT_INGREDS>", "<END_INGREDS>",
    "<BEGIN_INSTR>", "<NEXT_INSTR>", "<END_INSTR>"
]

# Load the data
def load_data(tokenized_file):
    with open(tokenized_file, 'r', encoding="utf-8") as file:
        data = file.readlines()
    return data

# Split the data
def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    return train_data, test_data

# Define the Dataset class
class RecipeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx].strip()
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": inputs.input_ids.squeeze()
        }

# Initialize the tokenizer
def initialize_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer.add_tokens(custom_tokens)
    tokenizer.save_pretrained('./tokenizer_with_custom_tokens')
    return tokenizer

# Initialize the model
def initialize_model(tokenizer):
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.resize_token_embeddings(len(tokenizer))
    return model

# Create DataLoaders with smaller batch size
def create_dataloaders(train_data, test_data, tokenizer, batch_size=4):  # Reduced batch size
    train_dataset = RecipeDataset(train_data, tokenizer)
    test_dataset = RecipeDataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader

# Training with mixed precision (fp16)
def train_model(model, tokenizer, train_dataloader, test_dataloader, output_dir='./t5_recipe_model'):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # Reduced batch size
        per_device_eval_batch_size=4,   # Reduced batch size
        evaluation_strategy="steps",
        logging_steps=500,
        save_steps=1000,
        eval_steps=1000,
        num_train_epochs=3,
        save_total_limit=2,
        learning_rate=3e-4,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        fp16=True,  # Enable mixed precision training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=test_dataloader.dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# Main function
def main():
    # Load tokenized data
    tokenized_file = 'tokenized_recipes.txt'
    data = load_data(tokenized_file)

    # Split into train and test
    train_data, test_data = split_data(data)

    # Initialize tokenizer
    tokenizer = initialize_tokenizer()

    # Initialize model
    model = initialize_model(tokenizer)

    # Move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()

    # Create DataLoaders
    train_dataloader, test_dataloader = create_dataloaders(train_data, test_data, tokenizer)

    # Train the model
    train_model(model, tokenizer, train_dataloader, test_dataloader)

# Run the main function
if __name__ == "__main__":
    main()
