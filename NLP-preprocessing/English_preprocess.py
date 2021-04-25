import pandas as pd
import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

class English_preprocessing:
    def __init__ (self, nlp_object_disable=['parser', 'ner','pos']):
        # load spacy model
        self.nlp = spacy.load('en', disable=nlp_object_disable)

    #clean text with regex. custom_replacements list is empty by default
    def regex_cleaner(self, string:str, custom_replacements = []) -> str:    
        #list of tuples, where tuples are paterns to be replaced
        base_replacements = [
               (r'\\n',' '),                   # remove newline
               (r'(\s+)', ' ')                 # remove multiple spaces, created by the cleaning above 
              ]
        replacements = base_replacements + custom_replacements  
        for pat,repl in replacements:
            string = re.sub(pat, repl, string)
        return string

    def tokenize(self, doc:str) -> list:
        sentence = self.nlp(doc)
        return [token.text.lower() for token in sentence]

    def remove_stopwords(self, doc:str) -> list:
        # Create list of stopwords
        stop_words = spacy.lang.en.stop_words.STOP_WORDS  
        return [token for token in self.tokenize(doc) if token not in stop_words]

    def lemmatize(self, sentence: str) -> list:
        sentence = self.nlp(doc)
        return [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in sentence]

