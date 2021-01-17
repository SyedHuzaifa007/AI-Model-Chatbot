# Impoting Modules

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tenserflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAvergePooling1D
from tenserflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Loading the data from the json file

with open('intents.json') as file:
    data = json.load(file)

# Extracting the required data from the json file

training_sentences = []
training_labels = []
labels = []
responses = []

for i in data['intents']:
    for p in data['patterns']:
        training_sentences.append(p)
        training_labels.append(i['tag'])
    responses.append(i['responses'])

    if i['tag'] not in labels:
        labels.append(i['tag'])

        