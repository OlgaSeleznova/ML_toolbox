import pandas as pd
import numpy as np
import hebrew_tokenizer as tokenizer
import re
from yap_api import YapApi
from collections import defaultdict

class Hebrew_preprocessing:
    def __init__ (self, additional_replacements = []):
     
        #list of tuples, where tuples are paterns to be replaced
        base_replacements = [
               (r'(https.*$ )', ' '),           # remove links. here regex is used instead of Beautiful soup, because Beutiful soup doesn't work with Hebrew.
               (r'href="(.*?)"', ' '),
               (r'\\n?(https.*)\\n', ' '),
               (r'(https.*)\\n', ' '),          # remove html tags
               (r'(\&nbsp;)', ' '),             
               (r'(\\&quot;)', ' '),
               (r'\n',' '),                     # remove newline
               (r'(\\r)',' '),
               (r'[\:;&#=\{\}\(\)]',' '),       # remove special chatacters. [a-z0-9] won't work because of Hebrew
               (r'(\s+)', ' ')                 # remove multiple spaces, created by the above cleaning 
              ]
        self.replacements = base_replacements + additional_replacements
    
    def regex_replacement(self, string:str, custom_replacements = []) -> str:             
        for pat,repl in self.replacements + custom_replacements:
            string = re.sub(pat, repl, string)
        return string        
    
    def get_groups_tokenized(self, text:str)-> dict():
        tokens = tokenizer.tokenize(text)
        groups = defaultdict(list)
        for grp, token, token_num, (start_index, end_index) in tokens:
            group,tok = str(grp), str(token)
            if group == 'Groups.HEBREW':
                groups['tokens'].append(token)
            if group == 'Groups.PUNCTUATION':
                groups['punct'].append(token)
            if group == 'Groups.NUMBER':
                groups['numbers'].append(token)
        return groups  
    
    def tokenize(self, text:str)-> list():
        return self.get_groups_tokenized(text)['tokens']
    
    def punctuation(self, text:str)-> list():
        return self.get_groups_tokenized(text)['punct']
    
    def punctuation_count(self, text:str)-> list():
        return len(self.get_groups_tokenized(text)['punct'])
    
    def numbers(self, text:str)-> list():
        return self.get_groups_tokenized(text)['numbers']
    
    def numbers_count(self, text:str)-> list():
        return len(self.get_groups_tokenized(text)['numbers'])
    
    """
    Set up Hebrew Lematizer before the function. It need Go pre-installed. 
    Documentation: https://nlp.biu.ac.il/~rtsarfaty/onlp/hebrew/documentation
    IP of YAP server: ip = '127.0.0.1:8000'
    yap = YapApi()
    """
    def lemmatize(self, list_tokens:list) -> str:
        text = ' '.join(list_tokens)
        tokenized_text, segmented_text, lemmas, dep_tree, md_lattice, ma_lattice = yap.run(text, ip)
        return lemmas
    """
    Hebrew stopwords are loaded from "he_stopwords.txt" file. 
    """
    def count_stopwords(self,text:str, stopwords) -> int:
        tokens = text.split(' ')   
        stopw = []
        count = 0
        for w in stopwords:
            if w in tokens:
                for i in range(len(tokens)): 
                    if (w == tokens[i]): 
                        count = count + 1
                stopw.append(w)
        return count
        
    def remove_stopwords(self,text:str, stopwords) -> str:
        tokens = text.split(' ') 
        stopw = []
        for w in stopwords:
            while w in tokens:
                tokens.remove(w)
        return text

    def save_csv(self, dataFrame, name:str):
        dataFrame.to_csv(name+'.csv',index=False)

