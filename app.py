import keras
from flask import Flask, request, render_template, jsonify
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np


app = Flask(__name__)

model = keras.models.load_model(r'./dataset/fake_model.h5') #tf.keras.models.load(r'./dataset/fake_model.h5')

@app.route('/')
def home():
    return render_template('/fakenews.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    token = Tokenizer()
    #function to make prediction
    bad_seq = token.texts_to_sequences(text)
    bad_pad = pad_sequences(bad_seq, maxlen=25, dtype='int32', value=0)
    pred = model.predict(bad_pad)
    for i in range(len(pred)):
        if pred[i].item() > 0.5:
            rslt = 'Fake News'
        else:
            rslt = 'Real News'
    return render_template('fakenews.html', preds = f'It\'s {rslt}')

if __name__=='__main__':
    app.run(debug = True)