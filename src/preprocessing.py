import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
import re

stop_words = stopwords.words('english')


#Removing special character
import re
def remove_special(text):
    clean_text = re.sub(r"[^a-zA-Z]", " ", text)
    return clean_text

# Removing URL's
def remove_url(content):
    return re.sub(r'http\S+', '', content)

#Removing the stopwords from text
def remove_stopwords(content):
    clean_data = []
    for i in content.split():
        if i.strip().lower() not in stop_words and i.strip().lower().isalpha():
            clean_data.append(i.strip().lower())
    return " ".join(clean_data)

def remove_extra_white_spaces(text):
    pattern=r'\s+[a-zA-Z]\s+'
    without_space=re.sub(pattern=pattern,repl=" ",string=text)
    return without_space
    

# Expansion of english contractions
def contraction_expansion(content):
    content = re.sub(r"won\'t", "would not", content)
    content = re.sub(r"can\'t", "can not", content)
    content = re.sub(r"don\'t", "do not", content)
    content = re.sub(r"shouldn\'t", "should not", content)
    content = re.sub(r"needn\'t", "need not", content)
    content = re.sub(r"hasn\'t", "has not", content)
    content = re.sub(r"haven\'t", "have not", content)
    content = re.sub(r"weren\'t", "were not", content)
    content = re.sub(r"mightn\'t", "might not", content)
    content = re.sub(r"didn\'t", "did not", content)
    content = re.sub(r"n\'t", " not", content)
    return content

# Stemming 
def Stemmer(text):
    
    ps = PorterStemmer()
    return [ps.stem(word) for word in word_tokenize(text)]

#Data preprocessing
def data_cleaning(content):
    content = contraction_expansion(content)
    content = remove_extra_white_spaces(content)
    content = remove_url(content)
    content=remove_special(content)
    content = remove_stopwords(content)
    content=Stemmer(content) 
    content= ' '.join(content)
      
    return content

class DataCleaning(BaseEstimator,TransformerMixin):
    def __init__(self):
        print('calling--init--')
    def fit(self,X,y=None):
        print('calling fit')
        return self
    def transform(self, X,y=None):
        print('calling transform')
        X=X.apply(data_cleaning)
        return X

