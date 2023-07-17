import spacy
import json
import better_profanity

# Load the SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load the JSON dataset
with open("./prompts.json") as json_file:
    data = json.load(json_file)

# Function to check if a given text contains any profanity
def contains_profanity(text):
    return better_profanity.Profanity().contains_profanity(text)

# Function to process user input and return the bot's response
def get_bot_response(user_input):
    # Check for profanity in user input
    if contains_profanity(user_input):
        return "I'm sorry, but I cannot respond to inappropriate language."

    # Process user input using SpaCy NLP model
    doc = nlp(user_input)

    # Check if user input contains any of the keywords
    for token in doc:
        for section in data.values():
            if isinstance(section, list):
                for item in section:
                    for key, value in item.items():
                        if isinstance(value, str):
                            if token.text.lower() in value.lower():
                                return data["prompts"][token.text.lower()]

    # If no keyword match, return a default response
    return data["prompts"]["default"]

# Example usage
user_input = input("You: ")
while user_input.lower() != "exit":
    bot_response = get_bot_response(user_input)
    print("Bot:", bot_response)
    user_input = input("You: ")
