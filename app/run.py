import json
import plotly
import pandas as pd
import numpy as np
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    stop_words = stopwords.words('english')
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if (clean_tok.isalpha() and clean_tok not in stop_words):  # filtering out punctuation and stop words
            clean_tokens.append(clean_tok)

    return clean_tokens


class Text_Length_Extractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(lambda x: len(x)).values
        return pd.DataFrame(X_tagged)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('clean_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


def top_words(message):
    """bring top 10 words in a message
        Parameters
        ----------
        df : dataframe
         Returns
        -------
        top 10 words and number of accurences : list
        """
    all_words = message.apply(tokenize).agg(np.sum)  # tokenize and aggregate all words
    unique_words = set(all_words)
    dicts = {w: all_words.count(w) for w in unique_words}  # create a dictionary pair, word and number of occurence
    top10_dict = dict(Counter(dicts).most_common(10))  # top 10
    words = list(top10_dict.keys())
    count = list(top10_dict.values())

    return words, count


print(" Extracting the mostly used words from the messages, \n please wait... ")
words, word_counts = top_words(df["message"])


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # Top 10 categories distribution
    category_names = df.columns[4:].tolist()
    top_category_num = df[category_names].sum().sort_values(ascending=False)[:10]
    top_category_names = list(top_category_num.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=top_category_num.values,
                    marker={'color': top_category_num.values}
                )

            ],

            'layout': {
                'title': 'Top 10 Categories in Data Set',
                'yaxis': {
                    'title': "Number of Occurence"
                },
                'xaxis': {
                    'title': "Category Name"
                }

            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=word_counts,
                    marker={'color': word_counts}
                )

            ],

            'layout': {
                'title': '"Top 10 Frequent Used Words"',
                'yaxis': {
                    'title': "Number of Used"
                },
                'xaxis': {
                    'title': "Word in Messages"
                }

            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()