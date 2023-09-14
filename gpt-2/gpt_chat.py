from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = './output'  # Path to the fine-tuned model directory
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Define a function for generating responses
def generate_response(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
user_input = input('Write a question:  ')
response = generate_response(user_input)
print(response)