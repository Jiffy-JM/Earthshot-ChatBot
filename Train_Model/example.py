from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('Train_Model/trained_model')
tokenizer = GPT2Tokenizer.from_pretrained('Train_Model/trained_model')

# Adjust hyperparameters
model.config.update({"n_layer": 12, "n_head": 8, "num_train_epochs": 10})

# Generate a response from the chatbot
def get_chatbot_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    output = model.generate(input_ids, max_length=100,
                                        num_return_sequences=1,
                                        num_beams=5,
                                        temperature=0.8,
                                        top_k=100,
                                        top_p=0.7,
                                        no_repeat_ngram_size=2,
                                        early_stopping=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

'''
# Example usage
user_input = input("\n\nUser: ")
while user_input.lower() != 'exit':
    chatbot_response = get_chatbot_response(user_input)
    print("\n\nChatbot:", chatbot_response)
    user_input = input("User: ")
'''


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
    app.run(debug=True,host="0.0.0.0",port="80")