import spacy
import json

nlp = spacy.load("en_core_web_sm")

# Load input-output pairs from JSON file
with open("data.json", "r") as json_file:
    data = json.load(json_file)

def process_input(input_text):
    doc = nlp(input_text)
    # Extract entities, POS tags, or other information
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    pos_tags = [(token.text, token.pos_) for token in doc]
    return entities, pos_tags

def generate_response(input_text):
    for pair in data:
        if pair["input_text"] == input_text:
            return pair["output_text"]
    return "I'm sorry, I don't have information on that topic."

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    response = generate_response(user_input)
    print("Bot:", response)
