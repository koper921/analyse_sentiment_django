import ast

import dill as pickle
import pandas as pd
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

from train_model.preprocess import TextCleaner
from train_model.train import replace_label




def classify_sentence(model, data, w2id, train =False  ):
    stop_words = ['le', 'la']
    taille_voc = 10000
    max_len = 25

    #
    df = pd.DataFrame(ast.literal_eval(json.loads(json.dumps(data))))
    #df = data[data['text'].notna()]
    tc = TextCleaner(stop_words, taille_voc, w2id, train)
    df['clean_text_split'] = tc.transform(df['text'])
    df['tokenized_int'], w2id = tc.transform_index(list(df['clean_text_split']))

    to_predict = pad_sequences(df['tokenized_int'].values, maxlen=max_len, value=0)
    pred = model.predict(to_predict)
    return (np.argmax(pred, axis = 1))

def load_dict(path):
    with open(path, 'r') as j:
        contents = json.loads(j.read())

    return (contents)


def load_model_sent(path_json, path_h5):
    json_file = open(path_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(path_h5)
    return (model)