import spacy
import json

# Load spaCy model (you may need to download the model if not already done)
nlp = spacy.load("en_core_web_sm")

# Load JSON data
with open('./prompts.json') as f:
    data = json.load(f)

def process_user_input(user_input, prompts):
    user_input = user_input.lower()
    # Check if the user's input matches any predefined prompts
    for prompt, response in prompts.items():
        if prompt in user_input:
            return response

    # If no match is found, check the categories for more specific responses
    for category, category_data in data['categories'].items():
        if category_data is None:
            continue

        keywords = category_data['keywords']
        for keyword in keywords:
            if keyword in user_input:
                return category_data['response']

    # If no specific match is found, use the default response
    return prompts['default']

def main():
    print("Chatbot: Hello! I'm a chatbot. You can start a conversation. Type 'goodbye' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'goodbye':
            print("Chatbot: Goodbye! Have a nice day.")
            break
        
        # Process user input and get the response
        response = process_user_input(user_input, data['prompts'])
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
