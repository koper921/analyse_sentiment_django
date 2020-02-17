
import pandas as pd
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences

from train_model.preprocess import TextCleaner
from train_model.model import mult_conv

import json

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import metrics

def replace_label(lab):
    if lab == '4':
        return 1
    else :
        return 0


df = pd.read_csv('train_model/French-Sentiment-Analysis-Dataset/tweets.csv', encoding = "ISO-8859-1", header = None, names = ["polarity", "text"], sep =",")
df = df[df['text'].notna()]
w2id = {}
train =True
stop_words = ['le', 'la']
taille_voc = 10000
tc = TextCleaner( stop_words, taille_voc, w2id, train)
df['clean_text_split'] = tc.transform(df['text'])
df['tokenized_int'], w2id = tc.transform_index(list(df['clean_text_split'] ))
df['label'] = df['polarity'].apply(lambda x : replace_label(x))

X_train, X_valid, y_train, y_valid = train_test_split(df['tokenized_int'].values, df['label'].values, test_size=0.25, random_state=0)

max_len = 25
x_train = pad_sequences(X_train, maxlen=max_len, value=0)
x_valid = pad_sequences(X_valid, maxlen=max_len, value=0)

hot_y = np_utils.to_categorical(y_train)
hot_y_valid = np_utils.to_categorical(y_valid)


multconv = mult_conv(taille_voc, max_len)
multconv_history = multconv.fit(x_train, hot_y, validation_data=(x_valid, hot_y_valid), epochs=1, batch_size=100)

# serialize model to JSON
model_json = multconv.to_json()
with open("model.json", "w") as json_file:
    #json_file.write(model_json)
# serialize weights to HDF5
multconv.save_weights("model.h5")
print("Saved model to disk")


# save dictionnary of vocabulary

json_file = json.dumps(w2id)
f = open("w2id.json","w")
f.write(json_file)
f.close()