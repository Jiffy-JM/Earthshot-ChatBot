# Earthshot-ChatBot
Earthshot is a project to spread awareness to sustainable means of inspiring people to take Climate Action

### Create your Docker Eniroment [Docker](https://www.docker.com/)
To acheive running the training of data and chatbot intance you must create a Docker Image and run the enviroment. See below on the steps of running creating the Docker Image and Running your Container:
```docker build -t myproject .```

```docker run --name mycontainer -p 8080:80 myproject```

## Dependencies
To achieve a locally running AI chat model that can be fine-tuned with our own data, it was necessary to utilize the following libraries.

Spacy is used as an NLP in order to use the large model run the following command

```python -m spacy download en_core_web_lg```

### [Flask](https://pypi.org/project/Flask/)
As a "microframework," Flask allows us to use Python in web development. Flask is a solution for development, and will be replaced with an enterprise level WSGI solution for the production application. 
### [PyTorch](https://pytorch.org/get-started/locally/)
PyTorch is an open source machine learning framework that is required in tandem with the transformers library to perofrm the ML tasks required for this project.
### [🤗Transformers](https://pypi.org/project/transformers/)
This library is created and maintained by [Hugging Face](https://huggingface.co/) and provides an extensive selection of pretrained models that can be used for ML. The types of models range in their intended use cases, and we have identified that QA Models are best suited for this project.

## Usage // Development
To run the program locally, first you will need to ensure that all of the dependencies above are installed. Next, request a ```trained_model``` folder from a team member and place it within the ```Train_Model``` folder. </br></br>
Now that all is in order, navigate to the ```example.py``` file in the ```Train_Model``` folder, and run it. </br></br> With this program running, a server will be laucnhed on your computer via Flask. Go to your preferred localhost address (i.e. 127.0.0.1) and there you go! You have successfully launched the chatbot locally and can interact with it in-browser!</br></br>Note: if you are experiencing issues with the local server, ensure that the target port (80) is not in use.

