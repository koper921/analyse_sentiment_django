from collections import Counter
import re

from sklearn.base import TransformerMixin, BaseEstimator


class TextCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, stop, taille_voc, w2id, train):
        self.stop = stop
        self.taille_voc = taille_voc
        self.vocab_counter = Counter()
        self.w2id = w2id
        self.remove_stop = True
        self.train = train

    def remove_mentions(self, text):

        return re.sub(r'@\w+', '', text)

    def remove_urls(self, text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', text)

    def only_characters(self, text):
        return re.sub('[^a-zA-Z\s]', '', text)

    def remove_extra_spaces(self, text):
        text = re.sub("\s+", ' ', text)
        text = text.lstrip()
        return text.rstrip()

    def to_lower(self, text):
        return text.lower()

    def tokenize(self, text):
        return text.split()

    def remove_stopwords(self, text_split):
        return [t for t in text_split if t not in self.stop]

    def update_vocab_counter(self, text_split):
        for word in text_split:
            self.vocab_counter[word] += 1

    def w2id_function(self):
        vocab = sorted(self.vocab_counter, key=self.vocab_counter.get, reverse=True)
        self.w2id = {w: i for i, w in enumerate(vocab[:self.taille_voc])}
        self.w2id['unk'] = 0

    def transform_to_ids(self, text_split):
        return [self.w2id[w] if w in self.w2id else self.w2id['unk'] for w in text_split]

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):

        # clean
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.only_characters).apply(
            self.remove_extra_spaces).apply(self.to_lower)

        if self.remove_stop:
            return clean_X.apply(self.tokenize).apply(self.remove_stopwords)
        else:
            return clean_X.apply(self.tokenize)

    def transform_index(self, list_text_split):
        list_text_index = []

        if self.train:
            # counter
            for text_split in list_text_split:
                self.update_vocab_counter(text_split)
            # word2index
            self.w2id_function()

        # transform to index
        for text_split in list_text_split:
            list_text_index.append(self.transform_to_ids(text_split))

        return list_text_index, self.w2id


