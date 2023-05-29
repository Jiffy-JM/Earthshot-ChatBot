from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('Train_Model/trained_model')
tokenizer = GPT2Tokenizer.from_pretrained('Train_Model/trained_model')

# Adjust hyperparameters
model.config.update({"n_layer": 12, "n_head": 8, "num_train_epochs": 10})

# Generate a response from the chatbot
def get_chatbot_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
user_input = input("User: ")
while user_input.lower() != 'exit':
    chatbot_response = get_chatbot_response(user_input)
    print("Chatbot:", chatbot_response)
    user_input = input("User: ")
