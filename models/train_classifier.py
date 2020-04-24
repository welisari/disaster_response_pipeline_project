import sys
# import libraries
# for data accessing
import pandas as pd
from sqlalchemy import create_engine

# Natural language ToolKits
import re
import nltk

nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Model Selections
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score

import numpy as np
import time
import pickle


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('clean_messages', con=engine)
    X = df["message"]
    categories = df.columns[4:]
    y = df[categories]
    return X, y, categories


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    stop_words = stopwords.words('english')
    for tok in tokens:
        if (tok.isalpha() and tok not in stop_words):  # filtering our punctuation and stop words
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
    return clean_tokens


# Building a machine learning pipeline
# This machine pipeline should take in the message column as input and output classification \
# results on the other 36 categories in the dataset
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
        'clf__estimator__n_estimators': [10,50],
        'clf__estimator__min_samples_split': [2, 3, 4]
        # 'vect__ngram_range': [(1, 1)]
        }

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    report = []
    f1_scores = []
    for i, col in enumerate(category_names):
        y_pred = model.predict(X_test)
        y_true = y_test[col]
        f = classification_report(y_true, y_pred[:, i])
        score = f1_score(y_true, y_pred[:, i], average='weighted')
        report.append(f)
        f1_scores.append(score)

    avg_f1 = np.mean(f1_scores)
    return print("Avg weighted f1-score:{}".format(avg_f1))


def save_model(model, model_filepath):
    # Save to file in the current working directory
    # pkl_filename = "classifier.pkl"
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        start = time.process_time()
        print('Building model...')
        model = build_model()
        print(time.process_time() - start)

        print('Training model...')
        model.fit(X_train, y_train)
        print(time.process_time() - start)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)
        print(time.process_time() - start)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print(time.process_time() - start)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()