from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import json

# Load pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<pad>')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load and preprocess your training data
train_data = 'refined_training_data.json'

# Load JSON data
with open(train_data, 'r') as f:
    data = json.load(f)

# Extract text from JSON data
texts = [item for item in data['data']]

# Join the texts with newline separator
train_text = '\n'.join(texts)

# Save the preprocessed text to a file
with open('preprocessed_train.txt', 'w') as f:
    f.write(train_text)

# Prepare the data for training
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='preprocessed_train.txt',
    block_size=128,
    overwrite_cache=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir='./model',
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('Train_Model/trained_model')
tokenizer.save_pretrained('Train_Model/trained_model')
