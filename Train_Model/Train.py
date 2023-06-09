from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import json
# Load pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<pad>')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)


# Load and preprocess your training data
train_data = 'refined_training_data.json'

#load json data
with open (train_data, 'r') as f:
    data = json.load(f)

#extract text from JSON data
texts = [item for item in data['data']]

#tokenize the text
tokenized_texts = tokenizer(texts, truncation=True, padding=True)

# Prepare the data for training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# create the text dataset
dataset = TextDataset(tokenizer=tokenizer,
                      file_path=None,block_size=128,
                      tokenized_text=tokenized_texts['input_ids'],
                      data_collator=data_collator
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
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('Train_Model/trained_model')
tokenizer.save_pretrained('Train_Model/trained_model')
