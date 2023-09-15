import spacy
import json
import random

# Load the spaCy English model
nlp = spacy.load("en_core_web_lg")

# Load the JSON file containing the question-answer pairs
with open("./data.json", "r") as file:
    qa_pairs = json.load(file)

# List to store the selected prompts
selected_prompts = []

# Variable to store the user's context

def retQA():
    return qa_pairs
# Function to retrieve a random input prompt and its corresponding output text
def get_random_prompt():
   
    relevant_prompts = []

    if user_context:
        # Filter prompts based on similarity to user's context
        for qa_pair in qa_pairs:
            input_text = qa_pair["input_text"]
            similarity = nlp(user_context).similarity(nlp(input_text))
            if similarity >= 0.10:  # Adjust similarity threshold as needed
                relevant_prompts.append((input_text, qa_pair["output_text"]))

    if not relevant_prompts:
        # If no relevant prompts found, choose a random one
        random_prompt = random.choice(qa_pairs)
        selected_prompts.append(random_prompt)
        return random_prompt["input_text"], random_prompt["output_text"]
    else:
        # Choose a relevant prompt randomly
        random_prompt = random.choice(relevant_prompts)
        return random_prompt[0], random_prompt[1]

# Function to reset the chat session
def reset_chat_session():
  
    user_context = None
    # Clear the list of selected prompts and restore all prompts to the available list
    qa_pairs.extend(selected_prompts)
    selected_prompts.clear()

# Function to get associated output and three additional responses
def get_responses(input_text):
    global user_context
    user_context = input_text  # Update user's context with the provided input

    # Get associated output for the input
    output_text = None
    for qa_pair in qa_pairs:
        if input_text.lower().strip() in qa_pair["input_text"].lower():
            output_text = qa_pair["output_text"]
            break
    
    if output_text is None:
        output_text = "I'm sorry, I don't have a response for that."

    # Generate three additional random prompts as responses
    additional_responses = [get_random_prompt()[0] for _ in range(3)]

    print(output_text, additional_responses)
    return output_text, additional_responses

# sample usage
# get_responses('How does Earthshot ensure the accuracy of information?')