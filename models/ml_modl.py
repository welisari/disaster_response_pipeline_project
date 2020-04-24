# import libraries
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine


import nltk
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
#nltk.download(['punkt', 'wordnet'])
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer