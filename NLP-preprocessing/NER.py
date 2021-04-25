class NER:
    def __init__ (self, model_name:str):
        self.model_name = model_name
        if self.model_name == 'spacy':
            self.spacy = spacy.load("en_core_web_lg")
        if self.model_name == 'stanza':
            self.stanza = stanza.Pipeline('en') # initialize English neural pipeline
        if self.model_name == 'flair':
            self.flair = SequenceTagger.load('ner')
    
    # ============== extact all entity types (spacy)==================
    def spacy_extractor(self, post:str) -> list:
        return self.spacy(post).ents  
    # extract entities
    def spacy_extract_entities(self, post:str) -> str:
        return ' , '.join([ent.text for ent in self.spacy_extractor(post)])
    # extract labels
    def spacy_extract_labels(self, post:str) -> str:
        return ' , '.join([ent.label_ for ent in self.spacy_extractor(post)])      
        
    # ============== extact all entity types (stanza)==================
    def stanza_extractor(self, post:str) -> list:
        return self.stanza(post).ents  
    # extract entities
    def stanza_extract_entities(self, post:str) -> str:
        return ' , '.join([ent.text for ent in self.stanza_extractor(post)])    
    # extract labels
    def stanza_extract_labels(self, post:str) -> str:
        return ' , '.join([ent.type for ent in self.stanza_extractor(post)])

    # ============== extact all entity types (flair) ==================
    def flair_extractor(self, post:str) -> list:
        sent = Sentence(post)
        self.flair.predict(sent)  
        if sent is not None:
            sent = sent.to_dict(tag_type='ner')['entities'] 
            return sent       
    # extract entities
    def flair_extract_entities(self, post:str) -> str:
        entities = [ent['text'] for ent in self.flair_extractor(post)]
        return ' , '.join(entities)
    # extract labels
    def flair_extract_labels(self, post:str) -> str:
        labels = [ent['labels'] for ent in self.flair_extractor(post)]
        if len(labels) >0:
            return re.sub(r'[\[\]]','', ' , '.join(map(str,labels)))
        else:
            return ' '
    
    def model_entities(self, post:str) -> str:
        if self.model_name == 'spacy':
            return self.spacy_extract_entities(post)
        if self.model_name == 'stanza':
            return self.stanza_extract_entities(post)
        if self.model_name == 'flair':
            return self.flair_extract_entities(post)
    
    def model_labels(self, post:str) -> str:
        if self.model_name == 'spacy':
            return self.spacy_extract_labels(post)
        if self.model_name == 'stanza':
            return self.stanza_extract_labels(post)
        if self.model_name == 'flair':
            return self.flair_extract_labels(post)
            
    def get_entities(self, data:dict) -> dict:
        ner_data = defaultdict(list)
        #count how many posts of one user
        counter = Counter(data['id'].values())
        for user in range(len(counter)):
            #print out logs
            if len(counter) > 50:
                if user%20 == 0:
                    print(f"Processing user number {user}")
            else:
                print(f"Processing user number {user}")
            #    
            for i in range(counter.get(data['id'][user])):
                ner_data['index'].append(user)
                ner_data['user'].append(data['id'][i])
                ner_data['post'].append(data['posts'][i])
                ner_data[self.model_name +'_entities'].append(self.model_entities(data['posts'][i]))
                ner_data[self.model_name +'_labels'].append(self.model_labels(data['posts'][i]))
                    
            else:
                continue
        return ner_data, counter

