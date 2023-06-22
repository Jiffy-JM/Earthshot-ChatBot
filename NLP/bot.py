import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define a dictionary of prompts and their corresponding responses
prompts = {
    "hello": "Hello there!",
    "how are you": "I'm doing well, thank you!",
    "goodbye": "Goodbye! Have a nice day.",
    "default": "I'm sorry, I don't understand. Can you please rephrase?"
}

# Function to generate a response based on user input
def generate_response(user_input):
    # Process user input
    doc = nlp(user_input.lower())

    # Check if user input matches any prompt
    for prompt, response in prompts.items():
        if prompt in doc.text:
            return response

    # If no prompt matches, return the default response
    return prompts["default"]

# Main loop
while True:
    # Get user input
    user_input = input("User: ")

    # Generate and print bot response
    bot_response = generate_response(user_input)
    print("Bot:", bot_response)
