import sys
import pandas as pd
import string
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from joblib import dump, load

from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class PredictPublication:

    def __init__(self):
        self.df = None
        self.lemmatizer = None
        self.x = None
        self.y = None
        self.bow_transformer = None
        self.model = None
        self.processed_count = 0

    def train(self, train_csv, test_size=0.2):
        self.processed_count = 0
        nltk_download('wordnet')
        nltk_download('stopwords')

        self.df = pd.read_csv(train_csv)
        self.df['title_content'] = self.df['title'] + ' ' + self.df['content']
        self.x = self.df['title_content']
        self.y = self.df['publication']

        self.lemmatizer = WordNetLemmatizer()
        # enocde publications
        self.labelencoder = LabelEncoder()
        self.y = self.labelencoder.fit_transform(self.y)

        x_train, x_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=0.2,
                                                            random_state=12345)
        self.total_files_to_process = x_train.shape[0] + x_test.shape[0]

        print(f'x_train shape: {x_train.shape}\nx_test_shape: {x_test.shape}\n'
              f'y_train_shape: {y_train.shape}\ny_test_shape: {y_test.shape}\n')

        self.bow_transformer = CountVectorizer(analyzer=self.process_text)
        text_bow_train = self.bow_transformer.fit_transform(x_train)
        text_bow_test = self.bow_transformer.transform(x_test)
        print(f'bow_train_shape: {text_bow_train.shape}\n'
              f'bow_test_shape: {text_bow_test.shape}\n')
        # multinomial naive bayes model
        self.model = MultinomialNB()
        # train the model
        self.model = self.model.fit(text_bow_train, y_train)
        print(self.model.score(text_bow_train, y_train))
        # validation
        print(self.model.score(text_bow_test, y_test))
        test_predictions = self.model.predict(text_bow_test)
        print(classification_report(y_test, test_predictions))

    def save_model(self, bow_filename, model_filename, labelenc_filename):
        """ Save raw objects of model """
        dump(self.bow_transformer, bow_filename)
        dump(self.model, model_filename)
        dump(self.labelencoder, labelenc_filename)
        print('Saved bag of words, label encoding, and model')

    def load_model(self, bow_filename, model_filename, labelenc_filename):
        """ Load raw objects of model """
        self.bow_transformer = load(bow_filename)
        self.model = load(model_filename)
        self.labelencoder = load(labelenc_filename)
        print('Loaded bag of words, label encoding, and model')

    def eval(self, title, content):
        """ Evaluate model for new sample """
        self.total_files_to_process = 1
        self.processed_count = 0
        combined_text = title + ' ' + content
        text_bow = self.bow_transformer.transform([combined_text])
        prediction = self.model.predict(text_bow)
        print(list(self.labelencoder.classes_))
        return self.labelencoder.inverse_transform(prediction)[0]

    def save_pipeline(self, pkl_filename):
        pipeline = Pipeline([('vectorizer', self.bow_transformer),
                             ('labelenc', self.labelencoder),
                             ('nbclassifier', self.model)])
        dump(pipeline, pkl_filename, compress=9)

    def eval_pipeline(self, pkl_filename, title, content):
        """ Evaluate a sample via the pipeline model """
        nltk_download('wordnet')
        nltk_download('stopwords')
        pipeline = load(pkl_filename)
        self.bow_transformer = pipeline['vectorizer']
        self.model = pipeline['nbclassifier']
        self.labelencoder = pipeline['labelenc']
        self.labelencoder
        return self.eval(title, content)

    def process_text(self, text):
        """ Standardize our text input for word2vec process """
        # remove punctuation
        no_punc_chars = [] #[char for char in text if char not in string.punctuation]
        for char in text:
            if char not in string.punctuation:
                # remove some of the unicode chars
                num = ord(char)
                if num >= 0 and num <= 127:
                    no_punc_chars.append(char)

        
        no_punc_words = ''.join(no_punc_chars).split()
        # downcase
        no_punc_words = [word.lower() for word in no_punc_words]
        # group inflected forms of word (ie. lemmatize)
        lemmatized_words = [self.lemmatizer.lemmatize(word, pos="v")
                                for word in no_punc_words ]
        no_stop_words = [word for word in lemmatized_words
                            if word not in stopwords.words('english')]

        # print friendly progress count while training
        self.processed_count += 1
        if self.processed_count % 100 == 0:
            print(f"Processed {self.processed_count}"
                  f" files of {self.total_files_to_process}")
        return no_stop_words

    def view_world_clouds(self, df, num_wc):
        """ visualize word cloud for given content """
        for i in range(num_wc):
            print(self.x[i])
            print(self.df['publication'][i])
            wc = WordCloud().generate(self.x[i])
            plt.imshow(wc, interpolation='bilinear')
            plt.show()


if __name__ == '__main__':
    cargs = len(sys.argv)
    bow_filename = 'predpub_bow.joblist'
    model_filename = 'predpub_nb.joblist'
    labelenc_filename = 'predpub_le.joblist'
    pkl_filename = 'predpub.pkl'

    usage = 'predict_publication <train|eval> [eval_title] [eval_content]'
    if cargs == 2:
        if sys.argv[1] == 'train':
            predpub = PredictPublication()
            predpub.train('data/train.csv')
            predpub.save_pipeline(pkl_filename)
    elif cargs == 4:
        if sys.argv[1] == 'eval':
            predpub = PredictPublication()
            print(predpub.eval_pipeline(pkl_filename, sys.argv[2], sys.argv[3]))

    else:
        print(usage)
