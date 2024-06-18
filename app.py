from flask import Flask, render_template, request, jsonify
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
import json
import torch
import random

app = Flask(__name__)

# Load intents JSON
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Load trained model and associated data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load("data.pth")
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    global chat_log
    user_input = request.json['user_input']
    tokenized_input = tokenize(user_input)
    bow_input = bag_of_words(tokenized_input, all_words)

    # Convert bag of words to tensor
    bow_tensor = torch.tensor(bow_input, dtype=torch.float32).unsqueeze(0).to(device)

    # Get model prediction
    output = model(bow_tensor)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Generate response based on predicted tag
    response = generate_response(tag)

    return jsonify({'response': response})


def generate_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])


if __name__ == '__main__':
    app.run(debug=True)
