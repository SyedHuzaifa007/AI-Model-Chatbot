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

label_encoder = LabelEncoder()
label_encoder.fit(training_labels)
training_labels = label_encoder.transform(training_labels)

vocab = 1000
embed_dim = 16
maximum_length = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(number_of_words = vocab, oov_token = oov_token)
tokenizer.fit_on_text(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating = 'post', maxlen = maximum_length)

