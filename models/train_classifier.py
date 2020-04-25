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
# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

import numpy as np
import time
import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """Load  a database table and return values, labels and category names
        Parameters
        ----------
        database_filepath : string
            location of the database
        Returns
        -------
        X: numpy.ndarray
            The training data
        y: numpy.ndarray
            The training labels
        categories: list
         The labely names category names
        """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('clean_messages', con=engine)
    X = df["message"].values
    categories = df.columns[4:].tolist()
    y = df[categories].values
    return X, y, categories


# defining a customized feature class

class Text_Length_Extractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(lambda x: len(x)).values
        return pd.DataFrame(X_tagged)


def tokenize(text):
    """Tokenize text
        Parameters
        ----------
        text : string
            the text to tokenize
        Returns
        -------
        clean_tokens : list
            the tokens list
        """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)  # tokenize

    lemmatizer = WordNetLemmatizer()  # initiate Lemmatizer

    # Lemmatize and normalize and strip
    clean_tokens = []
    stop_words = stopwords.words('english')
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if (clean_tok.isalpha() and clean_tok not in stop_words):  # filtering out punctuation and stop words
            clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """Build and optimize model
        Parameters
        ----------
        None

        Returns
        -------
        model
        """
    # building a pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('text_len', Text_Length_Extractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # set parameters for grid search
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100, 200],
        #'clf__estimator__max_features': ['log2', 'sqrt','auto'],
        #'clf__estimator__criterion': ['entropy', 'gini'],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'text_len': 0.3},
            {'text_pipeline': 0.5, 'text_len': 1},
            {'text_pipeline': 0.8, 'text_len': 1}
        )
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """Evaluates and prints model performance
        Parameters
        ----------
        model : multiclassification model
        X_test: numpy.ndarray
            The test data
        y_test: numpy.ndarray
            The test labels
        category_names: list
         The category names
        Returns
        -------
        Average weighted f1-score
        """
    y_pred = model.predict(X_test)
    final_scores = []
    for i in range(y_test.shape[1]):
        # print(classification_report(y_test[:, i], y_pred[:, i]))
        f1 = f1_score(y_test[:, i], y_pred[:, i], average='weighted')
        precision = precision_score(y_test[:, i], y_pred[:, i], average='weighted')
        recall = recall_score(y_test[:, i], y_pred[:, i], average='weighted')
        scores = [f1, precision, recall]
        final_scores.append(scores)

    avg_final_scores = np.around(np.mean(final_scores, axis=0), 3)
    print("Avg weighted f1-score,Precision and Recall Scores are:{}".format(avg_final_scores))


def save_model(model, model_filepath):
    """Save model as a pickle file
       Parameters
       ----------
       model : multiclassification model
           The optimized classifier
       model_filepath : string
           location of the database
       Returns
       -------
       None
       """
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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()