
# this is so fuckin dumb lmaoooooo
# if we can make the training happen when the program starts but then generate answers on deamnd that would be better, but
#   tbh its fine as is.
# 

import torch
import transformers
import re
from transformers import BertTokenizer, BertForQuestionAnswering

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Example text data
'''
text = """
Earthshot is a non-profit organization. They are using video games and immersive experiences to inspire people to take action on climate change.
Current messaging on climate change is not working. Most climate messaging is about negative consequences on a geographic scale. This is not moving people to take action that is needed now.
Games can move people by making participation rewarding, for example, through retail rewards for qualifying actions, or by providing the gratification of constructive competition towards a common goal.
Bob Wyman is a retired partner at Latham & Watkins LLP, where he served as Global Chair of its Environment, Land & Resources Department and Global Co-Chair of its Air Quality and Climate Practice Group.
Mark Bernstein is a Pioneering leader in developing ideas, policies, and technologies for improving sustainability. He held influential positions in various organizations in government, academia, and finance.
Earthshot was founded by Bob Wyman and Mark Bernstein, two friends and collaborators who have been working together on cleantech projects for over a decade.
Earthshot is currently working on a series of games that will be used to educate people about climate change.
Games that Earthshot is developing will be available on smartphones, tablets, and gaming consoles.
"Earthshot Now" is a podcast series that features interviews with celebrities, athletes, scientists, technologists, policymakers, and thought leaders about climate change. The podcast is available on Spotify, Apple Podcasts, and Google Podcasts.
Earthshot is co-hosting a technology showcase with John Preston, former Director of Technology Development at MIT, featuring zero and low-carbon options for US infrastructure.
SamsonPR is providing Earthshot with a public relations plan to activate social, digital, and print media.
Earthshot is exploring developing cleantech platforms that could be used in game franchises, such as an electric car platform that can be added to games like Gran Turismo and Microsoft Forza.
Earthshot intends to develop its own characters and narratives for possible use in games and experiences.
Earthshot plans to develop interactive, and immersive experiences for museums, county fairs, and schools. These experiences would engage participants in problem solving that demonstrates both the impact of climate change and the prospects for using cleantech to decarbonize the economy.
Earthshot may have the opportunity to work with a film studio on the production of a film focused on multiplayer games. The story is centered around a female lead in a world with no animals.
"""

text = text.split('\n')
text = " ".join(text)
print(text) '''

text = """
Earthshot, a nonprofit organization, is using video games to inspire action on climate change. Their approach involves making participation rewarding and fostering constructive competition. Earthshot was founded by Bob Wyman and Mark Bernstein. Bob Wyman is a retired partner at Latham & Watkins LLP, where he served as Global Chair of its Environment, Land & Resources Department and Global Co-Chair of its Air Quality and Climate Practice Group. Mark Bernstein is a Pioneering leader in developing ideas, policies, and technologies for improving sustainability and held influential positions in various organizations in government, academia, and finance. They both have been collaborating on cleantech projects for over a decade. Currently, Earthshot is developing games for smartphones, tablets, and gaming consoles. In addition, they have launched the "Earthshot Now" podcast, featuring interviews with various experts on climate change. The podcast is available on Spotify, Apple Podcasts, and Google Podcasts. To showcase zero and low-carbon options for US infrastructure, Earthshot is co-hosting a technology showcase with John Preston, former Director of Technology Development at MIT. To raise awareness, Earthshot has enlisted the services of SamsonPR for their public relations efforts. In terms of future plans, Earthshot aims to integrate cleantech platforms into popular game franchises, create their own characters and narratives for games, and develop interactive experiences for museums, county fairs, and schools. Moreover, there is a potential opportunity for Earthshot to collaborate with a film studio on a multiplayer game-based film set in a world devoid of animals, with a female lead at the center of the story.
"""



while True:
    # ask question
    question = input('ask question: ')

    # if question is q then break
    if question == 'q':
        break

    # Tokenize the question and text
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors='pt')

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

    print("Question: ", question)
    print("Answer: ", answer)