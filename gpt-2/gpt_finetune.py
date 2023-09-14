from transformers import GPT2LMHeadModel, GPT2Tokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import json

# Load conversations from the JSON file
with open('./data.json', 'r', encoding='utf-8') as json_file:
    conversations = json.load(json_file)

# Set the paths to the text file and pre-trained GPT-2 model
dataset_path = './data.json'
model_name = 'gpt2'  # You can choose other GPT-2 variants as well

# Load the pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the padding token to EOS (end of sequence)
tokenizer.pad_token = tokenizer.eos_token

# Load your custom dataset using LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=dataset_path,
    block_size=128,  # Adjust this based on your data length
)

# Prepare data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./output',  # Directory to save the trained model
    overwrite_output_dir=True,
    num_train_epochs=10,  # Adjust as needed
    per_device_train_batch_size=8,  # Adjust batch size as needed
    save_steps=10_000,  # Save model every X steps
    save_total_limit=2,  # Limit number of saved models
    evaluation_strategy="steps",
    eval_steps=10_000,  # Evaluate every X steps
)

# Create Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model()

# Optionally, save the tokenizer as well
tokenizer.save_pretrained('./output')
