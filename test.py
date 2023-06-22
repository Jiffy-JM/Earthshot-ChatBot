import json

#load json data
with open ('./Train_Model/refined_training_data.json', 'r') as f:
    data = json.load(f)

#extract text from JSON data
texts = [item for item in data['train_data']]

print(texts)

# data_object = json.loads(data)
# for element in data_object['data']:
#   print (element)