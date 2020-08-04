#NLP (Natural Language Processing)
#NLP Data Cleaning
import numpy as np
from collections import Counter
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem import SnowballStemmer
import string
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline  import Pipeline, FeatureUnion, make_pipeline
stops = set(nltk.corpus.stopwords.words('english'))
print(stops) #prints out list of stop words (commonly used word that a search engine has been programmed to ignore)
corpus = ["Jeff stole my octopus sandwich.", 
    "'Help!' I sobbed, sandwichlessly.", 
    "'Drop the sandwiches!' said the sandwich police."]
#How do I turn a corpus of documents into a feature matrix?
#Words --> numbers?????
#Corpus: list of documents
    #Original text:
    [
    "Jeff stole my octopus sandwich.", 
    "'Help!' I sobbed, sandwichlessly.", 
    "'Drop the sandwiches!' said the sandwich police."
    ]
def our_tokenizer(doc, stops=None, stemmer=None): #convert text above into tokens (2d arr)
    doc = word_tokenize(doc.lower())
    tokens = [''.join([char for char in tok if char not in string.punctuation]) for tok in doc]
    tokens = [tok for tok in tokens if tok]
    if stops:
        tokens = [tok for tok in tokens if (tok not in stops)]
    if stemmer:
        tokens = [stemmer.stem(tok) for tok in tokens]
    return tokens
tokenized_docs = [our_tokenizer(doc) for doc in corpus] #turns text into below
    #Step 1 result: stardarize - lowercase, lose punction, split into tokens
    [
    ['jeff', 'stole', 'my', 'octopus', 'sandwich'],
    ['help', 'i', 'sobbed', 'sandwichlessly'],
    ['drop', 'the', 'sandwiches', 'said', 'the', 'sandwich', 'police']
    ]
stopwords = set(nltk.corpus.stopwords.words('english')) #none duplicates
'i' in stopwords 
tokenized_docs = [our_tokenizer(doc, stops=stopwords) for doc in corpus]
tokenized_docs
    #Step 2 result: removed stop words
    [
    ['jeff', 'stole', 'octopus', 'sandwich'],
    ['help', 'sobbed', 'sandwichlessly'],
    ['drop', 'sandwiches', 'said', 'sandwich', 'police']
    ]
tokenized_docs = [our_tokenizer(doc, stops=stopwords, stemmer=SnowballStemmer('english')) for doc in corpus]
tokenized_docs
    #Step 3 result: Stemming/Lemmatization - go back to the root of each word (sandwiches => sandwich)
    [
    ['jeff', 'stole', 'octopus', 'sandwich'],
    ['help', 'sobbed', 'sandwichlessly'],
    ['drop', u'sandwich', 'said', 'sandwich', 'police']
    ]
#
#Count Vectorizer, TFIDF
Dictionary: ['drop', 'help', 'jeff', 'octopus', 'police', 'said', 'sandwich', 'sandwichlessly', 'sobbed', 'stole']
#Amount of times the words appeared in dictionary
    ['jeff', 'stole', 'octopus', 'sandwich']#sentence
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1]#frequency in vector
    ['help', 'sobbed', 'sandwichlessly']
    [0, 1, 0, 0, 0, 0, 0, 1, 1, 0]
    ['drop', u'sandwich', 'said', 'sandwich', 'police']
    [1, 0, 0, 0, 1, 1, 2, 0, 0, 0]
#TF term frquency: (#_𝑜𝑓_𝑡𝑖𝑚𝑒𝑠_𝑤𝑜𝑟𝑑_𝑎𝑝𝑝𝑒𝑎𝑟𝑠_𝑖𝑛_𝑑𝑜𝑐𝑢𝑚𝑒𝑛𝑡) / (𝑡𝑜𝑡𝑎𝑙_#_𝑜𝑓_𝑤𝑜𝑟𝑑𝑠_𝑖𝑛_𝑑𝑜𝑐𝑢𝑚𝑒𝑛𝑡)
    ['jeff', 'stole', 'octopus', 'sandwich']
    [0, 0, 1/4, 1/4, 0, 0, 1/4, 0, 0, 1/4]
    ['help', 'sobbed', 'sandwichlessly']
    [0, 1/3, 0, 0, 0, 0, 0, 1/3, 1/3, 0]
    ['drop', u'sandwich', 'said', 'sandwich', 'police']
    [1/5, 0, 0, 0, 1/5, 1/5, 2/5, 0, 0, 0]
#Document frequency: (#_𝑜𝑓_𝑑𝑜𝑐𝑢𝑚𝑒𝑛𝑡𝑠_𝑐𝑜𝑛𝑡𝑎𝑖𝑛𝑖𝑛𝑔_𝑤𝑜𝑟𝑑) / (𝑡𝑜𝑡𝑎𝑙_#_𝑜𝑓_𝑑𝑜𝑐𝑢𝑚𝑒𝑛𝑡𝑠)
    Vocabulary: ['drop', 'help', 'jeff', 'octopus', 'police', 'said', 'sandwich', 'sandwichlessly', 'sobbed', 'stole']
    Document frequency for each word: [1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 2/3, 1/3, 1/3, 1/3]
#IDF Inverse document frequency: log((𝑡𝑜𝑡𝑎𝑙_#_𝑜𝑓_𝑑𝑜𝑐𝑢𝑚𝑒𝑛𝑡𝑠) / (#_𝑜𝑓_𝑑𝑜𝑐𝑢𝑚𝑒𝑛𝑡𝑠_𝑐𝑜𝑛𝑡𝑎𝑖𝑛𝑖𝑛𝑔_𝑤𝑜𝑟𝑑))
    Vocabulary:['drop', 'help', 'jeff', 'octopus', 'police', 'said', 'sandwich', 'sandwichlessly', 'sobbed', 'stole']
    IDF for each word:[1.099, 1.099, 1.099, 1.099, 1.099, 1.099, 0.405, 1.099, 1.099, 1.099]
#TFIDF = TF*IDF
    ['jeff', 'stole', 'octopus', 'sandwich']
    [0, 0, 0.275, 0.275, 0, 0, 0.101, 0, 0, 0.275]
    ['help', 'sobbed', 'sandwichlessly']
    [0, 0.366, 0, 0, 0, 0, 0, 0.366, 0.366, 0]
    ['drop', u'sandwich', 'said', 'sandwich', 'police']
    [0.22, 0, 0, 0, 0.22, 0.22, 0.162, 0, 0, 0]
#similarity measure
cosine_similarity([[0, 0, 0.275, 0.275, 0, 0, 0.101, 0, 0, 0.275],  [0.22, 0, 0, 0, 0.22, 0.22, 0.162, 0, 0, 0]]) #output 2*2 arr ([[ 1.,0.081],[ 0.081,1.]])
cosine_similarity([[0, 0.366, 0, 0, 0, 0, 0, 0.366, 0.366, 0],  [0.22, 0, 0, 0, 0.22, 0.22, 0.162, 0, 0, 0]]) #output 2*2 arr ([[ 1.,0.],[ 0.,1.]])
#
#Spam data: revisit spam ham example 
df= pd.read_table('data/SMSSpamCollection', header=None) #read
df.columns=['spam', 'msg'] #assign headers
stopwords_set=set(stopwords) #stopwords set
punctuation_set=set(string.punctuation) #punctuation set
df['msg_cleaned']= df.msg.apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords_set \
                                                   and word not in punctuation_set])) #clean the text from the two sets' content
str1='Go until jurong point, crazy'.split() #explain above: this splits a string
' '.join(str1) #and this joins the seperate words back together with a space in between
df['msg_cleaned']= df.msg_cleaned.str.lower() #lower case
count_vect= CountVectorizer()
X= count_vect.fit_transform(df.msg_cleaned) #passing the whole column. X has shape (5572,8703)
y=df.spam
X_train, X_test, y_train, y_test= train_test_split(X,y)
lg= LogisticRegression()
lg.fit(X_train,y_train)
y_pred=lg.predict(X_test)
lg.score(X_test,y_test)
confusion_matrix(y_test, y_pred)
#
#Tweak model with spam data
## try tfidf  
tfidf= TfidfVectorizer()  
X= tfidf.fit_transform(df.msg_cleaned)
y=df.spam 
X_train, X_test, y_train, y_test= train_test_split(X,y)  
## try random forest 
rf= RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
rf.score(X_test,y_test)
confusion_matrix(y_test, y_pred)   
#try gradient boost 
gb= GradientBoostingClassifier()
gb.fit(X_train,y_train)
y_pred=gb.predict(X_test)
gb.score(X_test,y_test)
confusion_matrix(y_test, y_pred)
# Try tfidf with bigrams & trigrams 
tfidf=TfidfVectorizer(ngram_range=(1,3)) 
X= tfidf.fit_transform(df.msg_cleaned)
y=df.spam
X_train, X_test, y_train, y_test= train_test_split(X,y)
#try gradient boost 
gb= GradientBoostingClassifier()
gb.fit(X_train,y_train)
y_pred=gb.predict(X_test)
gb.score(X_test,y_test)
confusion_matrix(y_test, y_pred) 
#
tfidf=TfidfVectorizer()
X=tfidf.fit_transform(df.msg_cleaned)
y=df.spam
X_train, X_test, y_train, y_test=  train_test_split(X,y)
lg= LogisticRegression()
lg.fit(X_train,y_train)
y_pred=lg.predict(X_test)
lg.score(X_test,y_test)
confusion_matrix(y_test, y_pred) 
#
#Pipeline with Spam data (streamline and reuse code)
pipeline= Pipeline([('countvect', CountVectorizer(stop_words=stopwords_set)),\
                    #('tfidf', TfidfVectorizer(stop_words=stopwords_set)),\ #can easily change this to find the best combination
                    ('lg',  LogisticRegression())])
X=df.msg_cleaned #note we are passing the cleaned msg to the pipeline 
y=df.spam
X_train, X_test, y_train, y_test= train_test_split(X,y) 
pipeline.fit(X_train, y_train) 
y_pred= pipeline.predict(X_test)
print(pipeline.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))

pipeline= Pipeline([#('countvect', CountVectorizer(stop_words=stopwords_set)),\
                    ('countvect', CountVectorizer(stop_words=stopwords_set)),\
                    ('rf',  RandomForestClassifier())])
X=df.msg_cleaned #note we are passing the cleaned msg to the pipeline 
y=df.spam
X_train, X_test, y_train, y_test= train_test_split(X,y) 
pipeline.fit(X_train, y_train) 
y_pred= pipeline.predict(X_test)
print pipeline.score(X_test, y_test)
print confusion_matrix(y_test, y_pred)  
# the best one so far!

pipeline= Pipeline([#('countvect', CountVectorizer(stop_words=stopwords_set)),\
                    ('countvect', CountVectorizer(stop_words=stopwords_set, ngram_range=(1,3))),\
                    ('rf',  RandomForestClassifier())])
X=df.msg_cleaned #note we are passing the cleaned msg to the pipeline 
y=df.spam
X_train, X_test, y_train, y_test= train_test_split(X,y) 
pipeline.fit(X_train, y_train) 
y_pred= pipeline.predict(X_test)
print pipeline.score(X_test, y_test)
print confusion_matrix(y_test, y_pred) 