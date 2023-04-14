from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, LSTM, Dense, GlobalAveragePooling1D, Flatten
from keras.models import Model
import matplotlib.pyplot as plt
import random
import string
from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from flask import redirect
from bs4 import BeautifulSoup
import requests
from urllib.parse import quote

app = Flask(__name__)

with open('intents.json') as content:
  data1 = json.load(content)

tags = []
inputs = []
responses = {}

for intent in data1['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])

data = pd.DataFrame({"inputs":inputs,"tags":tags})

data = data.sample(frac=1)

data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data ['inputs'].apply(lambda wrd: ''.join(wrd))

#tokenization
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
#apply padding
from keras_preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)

#encoding the outputs
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])


input_shape = x_train.shape[1]

#define vocabulary
vocabulary = len(tokenizer.word_index)
print("number of unique words: ",vocabulary)
output_length = le.classes_.shape[0]
print("output length: ", output_length)


#creating model

i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(i)
x = LSTM(10,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)

#compiling the model

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

#training the model
train = model.fit(x_train, y_train, epochs=200)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prediction_input = request.form['input']
        texts_p = []

        # removing punctuation and converting to lowercase
        prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
        prediction_input = ''.join(prediction_input)
        texts_p.append(prediction_input)

        # tokenizing and padding
        prediction_input = tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input],input_shape)

        # getting output from model
        output = model.predict(prediction_input)
        output = output.argmax()

        # finding the right tag and predicting
        response_tag = le.inverse_transform([output])[0]
        result = random.choice(responses[response_tag])

        # create Lazada search link
        lazada_search_url = f"https://www.lazada.com.ph/catalog/?q={quote(response_tag)}"

        return render_template('index.html', result=result,lazada_search_url=lazada_search_url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()