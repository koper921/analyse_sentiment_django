from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, load_model
from keras.layers.embeddings import Embedding
from keras.layers import Flatten, Dense, Dropout, Convolution1D, MaxPooling1D, SpatialDropout1D, Input
from keras.layers import GlobalMaxPooling1D, concatenate, LSTM, Bidirectional, SimpleRNN, GRU
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def mult_conv(taille_voc, max_len):
    graph_in = Input(shape=(taille_voc, 50))

    convs = []
    for filter_size in range(2, 5):
        x = Convolution1D(64, filter_size, padding='same', activation='relu')(graph_in)
        convs.append(x)

    graph_out = concatenate(convs, axis=1)
    graph_out = GlobalMaxPooling1D()(graph_out)
    graph = Model(graph_in, graph_out)

    model = Sequential([Embedding(taille_voc, 50, input_length=max_len),
                        graph,
                        Dropout(0.5),
                        Dense(25, activation='relu'),
                        Dense(2, activation='sigmoid')])

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    return model