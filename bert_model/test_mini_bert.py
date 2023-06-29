import torch
import transformers
import re
import spacy
from transformers import BertTokenizer, BertForQuestionAnswering
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


# topic datasets 
default_dataset = """
Earthshot, a nonprofit organization, is using video games to inspire action on climate change. Their approach involves making participation rewarding and fostering constructive competition. Earthshot was founded by Bob Wyman and Mark Bernstein. Bob Wyman is a retired partner at Latham & Watkins LLP, where he served as Global Chair of its Environment, Land & Resources Department and Global Co-Chair of its Air Quality and Climate Practice Group. Mark Bernstein is a Pioneering leader in developing ideas, policies, and technologies for improving sustainability and held influential positions in various organizations in government, academia, and finance. They both have been collaborating on cleantech projects for over a decade. Currently, Earthshot is developing games for smartphones, tablets, and gaming consoles.
In addition, they have launched the "Earthshot Now" podcast, featuring interviews with various experts on climate change. The podcast is available on Spotify, Apple Podcasts, and Google Podcasts. To showcase zero and low-carbon options for US infrastructure, Earthshot is co-hosting a technology showcase with John Preston, former Director of Technology Development at MIT. To raise awareness, Earthshot has enlisted the services of SamsonPR for their public relations efforts. In terms of future plans, Earthshot aims to integrate cleantech platforms into popular game franchises, create their own characters and narratives for games, and develop interactive experiences for museums, county fairs, and schools. Moreover, there is a potential opportunity for Earthshot to collaborate with a film studio on a multiplayer game-based film set in a world devoid of animals, with a female lead at the center of the story.
"""

gaming_dataset = '''
Earthshot, a non-profit organization, is using video games and immersive experiences as a means to inspire people to take action on climate change. They believe that current messaging on climate change is not effective and that people are more likely to act when they see a positive vision of the future. Earthshot aims to create innovative games that people will want to play, where they can experience new technologies and their exciting opportunities. Through these games, individuals can see how their own actions can improve their quality of life while tackling climate change. Earthshot also believes that games can motivate people to take action by offering economic rewards for participating in qualifying actions or by providing the gratification of constructive competition towards a common goal.
Earthshot recognizes the power of games and immersive experiences to engage people and educate them about climate change. They are actively involved in the development of a series of games that will be released in the future. To expand their reach and impact, Earthshot is seeking partnerships with other organizations. Their efforts include the creation of hand-held games that aim to educate and inspire individuals to take action on climate change. Additionally, Earthshot is exploring the possibility of integrating cleantech platforms into existing game franchises, such as racing games, to promote eco-friendly practices.
In terms of collaborations, Earthshot may have the opportunity to work with a new film studio on a project that involves a multi-player game where players build their own wild animal park. Earthshot would provide guidance on environmental aspects and contribute to game development. Earthshot emphasizes the importance of equity and inclusivity in their work. They are actively engaging with diverse communities and organizations like First Star and Black Girls CODE to ensure that their products and initiatives benefit people of all backgrounds, promoting racial and gender equity in the fight against climate change.
Earthshot believes that video games can be a powerful tool for education and advocacy. The organization is currently developing two initial game concepts: "Electro Venture," a car-based adventure game that aims to get players excited about electric vehicles, and "Green City," a city-building game that teaches players about the importance of sustainability and how to build a more sustainable city. Earthshot's goal is to create a movement of gamers who are passionate about climate change and committed to taking action to address it.
'''

bob_and_mark_dataset = '''
Bob Wyman is a retired partner at Latham & Watkins LLP, has a background in environmental law and has worked on innovative strategies for reducing environmental impacts and advancing clean energy and transportation technologies.
Mark Bernstein has been a driving force in sustainability, focusing on influencing consumer behavior in areas like energy efficiency and recycling.
The founders of Earthshot are Bob Wyman and Mark Bernstein.
Together with industry experts like Academy Award-winning visual effects artist Rob Legato and augmented reality specialists Magnopus, Wyman and Bernstein co-founded Earthshot. This organization aims to inspire individuals to take action on climate change by creating positive, educational, and engaging experiences. They believe that the current negative-focused climate change messaging is ineffective, and instead, they emphasize the power of presenting a positive vision of the future.
The organization's board of directors and team consists of experts from diverse industries such as entertainment, gaming, climate policy, sustainability, and technology.
Earthshot is actively working on various initiatives, including the development of hand-held games, podcasts, technology showcases, public relations, and social networking campaigns. In the long term, they have plans to create museum-quality interactive and immersive experiences for venues like museums, county fairs, and schools. To support their work, Earthshot relies on funding from in-kind contributions, individual donations, and grants, while actively promoting racial and gender equity in their collaborations and initiatives.
Earthshot has additional goals, including the development of cleantech platforms for existing game franchises, the creation of unique character and narrative intellectual property, and potential collaborations on a film centered around the multi-player game "Norah's Arc."
Overall, Bob Wyman and Mark Bernstein play crucial roles in Earthshot's mission to inspire climate action through innovative gaming experiences
'''

investing_dataset = '''
Earthshot, as a non-profit organization, is funded through a combination of in-kind contributions, individual donations, and grants. It highlights that Earthshot has received significant in-kind contributions from a variety of technology firms, programmers, game developers, and legal and professional service advisors. Additionally, Earthshot has received individual donations from around 40% of the approximately 120 people who have been contacted for this purpose.
Earthshot is actively working on expanding its reach and fundraising efforts. It states that the organization is building a significantly larger mailing list in preparation for an upcoming individual fundraising campaign scheduled for later in the year. Furthermore, Earthshot is seeking grants from foundations and other organizations that show an interest in supporting its work.
'''

team_dataset = '''
Mark Bernstein is the president and co-founder of Earthshot.
Bob Wyman is the Co-founder and a Board Chair of Earthshot.
The Earthshot team includes: Nikki Buffa, partner at Latham and Watkins; Manuel Grace, Associate General Counsel at Disney; Jamar Graham, Product Manager at PlayerWON; Lauren Graham, Founder of Velvet Frame; Bruce Garfield, Founder of Garfield Agency; Nikhil Jain, Co-Founder of Oben Inc. and serial entrepeneur; Rob Legato, Triple Academy award winning special effects expert; Ariella Lehrer, CEO of Legacy Games; Kenny Leu, Actor and Entrepeneur; Clayton Munnings, evironmental economist and consultant; Vickie Patton, evnironmental defense Fund's General Counsel; John Preston, Founder of TEM Capital; Jerry Prochazka, CEO at Ganymede Games; Mike Vandenbergh, Professor of Law and Director of Climate Change Research at Vanderbilt University; Sean Watson, Innovation Strategist; Belinda Smith Walker, Founding Board Member of New Villiage Girls Academy;
'''

'''
while True:
    # ask question
    question = input('\n\nQuestion: ')

    # if question is q then break
    if question == 'q':
        break
    
    dataset = ''
    lowerQ = question.lower()
    # FILTER THE QUESTION FOR KEYWORDS TO CHOOSE DATASET
    if 'bob' in lowerQ or 'wyman' in lowerQ or 'mark' in lowerQ or 'bernstein' in lowerQ:
        dataset = bob_and_mark_dataset
        print('\nbob and mark')
    elif 'invest' in lowerQ or 'financ' in lowerQ:
        dataset = investing_dataset
        print('\ninvesting')
    elif 'game' in lowerQ or 'gaming' in lowerQ:
        dataset = gaming_dataset
        print('\ngaming')
    else:
        dataset = default_dataset
        print('\ndefault')
    
    # join lines in set
    dataset = dataset.split('\n')
    dataset = " ".join(dataset)

    # Tokenize the question and text
    inputs = tokenizer.encode_plus(question, dataset, add_special_tokens=True, return_tensors='pt')

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # get start and end scores
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the most probable answer
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    all_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
    answer = ' '.join(all_tokens[start_index: end_index + 1])
    answer = re.sub(r"##", "", answer)

    
    print("Answer: ", answer)'''

nlp = spacy.load("en_core_web_sm")

# Define a dictionary of prompts and their corresponding responses
greetings = {
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
    for prompt, response in greetings.items():
        if prompt in doc.text:
            return response

    # If no prompt matches, return the default response
    return greetings["default"]


# return the response from the http request
def get_chatbot_response(question):

    response = generate_response(question)
    
    if response != greetings["default"]:
        return response




    dataset = ''
    lowerQ = question.lower()
    # FILTER THE QUESTION FOR KEYWORDS TO CHOOSE DATASET
    if 'bob' in lowerQ or 'wyman' in lowerQ or 'mark' in lowerQ or 'bernstein' in lowerQ or 'board' in lowerQ:
        dataset = bob_and_mark_dataset
        print('\nbob and mark')
    elif 'invest' in lowerQ or 'financ' in lowerQ or 'money' in lowerQ or 'capital' in lowerQ or 'equity' in lowerQ:
        dataset = investing_dataset
        print('\ninvesting')
    elif 'game' in lowerQ or 'gaming' in lowerQ or 'green city' in lowerQ or 'electro ave' in lowerQ or 'develop' in lowerQ:
        dataset = gaming_dataset
        print('\ngaming')
    elif 'team' in lowerQ or 'president' in lowerQ or 'co-' in lowerQ:
        dataset = team_dataset
        print('\nteam')
    else:
        dataset = default_dataset
        print('\ndefault')
    
    # join lines in set
    dataset = dataset.split('\n')
    dataset = " ".join(dataset)

    # Tokenize the question and text
    inputs = tokenizer.encode_plus(question, dataset, add_special_tokens=True, return_tensors='pt')

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # get start and end scores
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the most probable answer
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    all_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
    
    # format the tokens
    answer = ' '.join(all_tokens[start_index: end_index + 1])
    answer = re.sub(r"##", "", answer)
    

    # Check if the answer is empty or invalid
    if not answer.strip() or len(answer.split()) == 1 or answer.strip() == ".":
        return greetings["default"]
    
    # Check if the answer starts with undesired patterns
    undesired_patterns = ["[cls]", "[sep]"]
    if any(answer.startswith(pattern) for pattern in undesired_patterns):
        return greetings["default"]

    # Format the BERT response with capitalized first letters and sentence-ending periods
    formatted_answer = answer.capitalize()
    formatted_answer = re.sub(r"(\w)([.?!])", r"\1\2 ", formatted_answer)
    formatted_answer = re.sub(r"\s+", " ", formatted_answer)
    formatted_answer = formatted_answer.strip() + "."

    return formatted_answer

#flask request and response handling
app = Flask(__name__, template_folder='templates')
CORS(app) #enable Cross Origin Resource Sharing (CORS)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/process-request', methods=['GET','POST'])
def process_request():
    if(request.method == 'POST'):

        data = request.json

        #process input
        chatbot_res = {'message' : get_chatbot_response(data)}

        return jsonify(chatbot_res)
    
    return 'method not allowed', 405

    

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port="8000")