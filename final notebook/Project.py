# importing required python libraries 
import nltk
import numpy as np
import os
import pandas as pd
import xgboost as xgb

from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB
from sklearn import svm

# extracting testing and training data from csv file using pandas
train_data = pd.read_csv('train.csv')
given_test_data = pd.read_csv('test.csv')

# creating a dictionary for special tokens
SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}


# defining a function clean to clean the data for easier modeling
def clean(text, stem_words=True):
    import re
    from string import punctuation
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords
    
    def pad_str(s):
        return ' '+s+' '
    
    if pd.isnull(text):
        return ''

    # stops = set(stopwords.words("english"))
    # Clean the text, with the option to stem words.
    
    # Empty question
    if type(text) != str or text == '':
        return ''

    # Clean the text
    text = re.sub("\'s", " ", text) 
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)

    # add padding to punctuations and special chars, we still need them later
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)

    # replace non-ascii word with special word
    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']), text)
    
    # indian dollar
    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)
    
    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text) 
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)
    
    # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with
    # word "number"
    text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation]).lower()

    # Return a list of words
    return text


# applying clean function on training data
train_data['question_text'] = train_data['question_text'].apply(clean)

# split to train and val
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=2018)

# split training data to train and test data
train_data, test_data = train_test_split(train_data, test_size=float(1.0/8), random_state=2018)

# storing question_text and target columns in different lists
train_text = train_data['question_text']
valid_text = val_data['question_text']
test_text = test_data['question_text']
train_target = train_data['target']
valid_target = val_data['target']
test_target = test_data['target']
all_text = train_text.append(train_text)

# applying TFIDF and count vectorization to question text and target values for each training and testing data
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(all_text)

train_text_features_tf = tfidf_vectorizer.transform(train_text)
test_text_features_tf = tfidf_vectorizer.transform(test_text)

# using kfold technique for fitting and predicting using Logistic Regression
kfold = KFold(n_splits=5, shuffle=True, random_state=2018)
test_preds = 0
oof_preds = np.zeros([train_data.shape[0],])

for i, (train_idx,valid_idx) in enumerate(kfold.split(train_data)):
    x_train, x_valid = train_text_features_tf[train_idx,:], train_text_features_tf[valid_idx,:]
    y_train, y_valid = train_target[train_idx], train_target[valid_idx]
    classifier1 = LogisticRegression()
    print('fitting.......')
    classifier1.fit(x_train,y_train)
    print('predicting......')
    print('\n')
    oof_preds[valid_idx] = classifier1.predict_proba(x_valid)[:,1]
    test_preds += 0.2*classifier1.predict_proba(test_text_features_tf)[:,1]

# training data F1 score
pred_train = (oof_preds > .25).astype(np.int)
f1_score(train_target, pred_train) 

# training data accuracy score
accuracy_score(train_target, pred_train)

# validating data F1 score
pred_valid = (oof_preds > .25).astype(np.int)
f1_score(valid_target, pred_valid) 

# validating data accuracy score
accuracy_score(valid_target, pred_valid)

# testing data F1 score
pred_test = (test_preds > .25).astype(np.int)
f1_score(test_target, pred_test)

# testing data accuracy score
accuracy_score(test_target, pred_test)

# using kfold technique for fitting and predicting using Naive Bayes
kfold = KFold(n_splits=5, shuffle=True, random_state=2018)
test_preds = 0
oof_preds = np.zeros([train_data.shape[0],])

for i, (train_idx,valid_idx) in enumerate(kfold.split(train_data)):
    x_train, x_valid = train_text_features_cv[train_idx,:], train_text_features_cv[valid_idx,:]
    y_train, y_valid = train_target[train_idx], train_target[valid_idx]
    classifier2 = MultinomialNB()
    print('fitting.......')
    classifier2.fit(x_train,y_train)
    print('predicting......')
    print('\n')
    oof_preds[valid_idx] = classifier2.predict_proba(x_valid)[:,1]
    test_preds += 0.2*classifier2.predict_proba(test_text_features_cv)[:,1]

# training data F1 score
pred_train = (oof_preds > .25).astype(np.int)
f1_score(train_target, pred_train) 

# training data accuracy score
accuracy_score(train_target, pred_train)

# validating data F1 score
pred_valid = (oof_preds > .25).astype(np.int)
f1_score(valid_target, pred_valid) 

# validating data accuracy score
accuracy_score(valid_target, pred_valid)

# testing data F1 score
pred_test = (test_preds > .25).astype(np.int)
f1_score(test_target, pred_test)

# testing data accuracy score
accuracy_score(test_target, pred_test)

# using kfold technique for fitting and predicting using XGBoost
kfold = KFold(n_splits=5, shuffle=True, random_state=2018)
test_preds = 0
oof_preds = np.zeros([train_data.shape[0],])

for i, (train_idx,valid_idx) in enumerate(kfold.split(train_data)):
    x_train, x_valid = train_text_features_tf[train_idx,:], train_text_features_tf[valid_idx,:]
    y_train, y_valid = train_target[train_idx], train_target[valid_idx]
    classifier3 = xgb.XGBClassifier()
    print('fitting.......')
    classifier3.fit(x_train,y_train)
    print('predicting......')
    print('\n')
    oof_preds[valid_idx] = classifier3.predict_proba(x_valid)[:,1]
    test_preds += 0.2*classifier3.predict_proba(test_text_features_tf)[:,1]

# training data F1 score
pred_train = (oof_preds > .25).astype(np.int)
f1_score(train_target, pred_train) 

# training data accuracy score
accuracy_score(train_target, pred_train)

# validating data F1 score
pred_valid = (oof_preds > .25).astype(np.int)
f1_score(valid_target, pred_valid) 

# validating data accuracy score
accuracy_score(valid_target, pred_valid)

# testing data F1 score
pred_test = (test_preds > .25).astype(np.int)
f1_score(test_target, pred_test)

# testing data accuracy score
accuracy_score(test_target, pred_test)

# using kfold technique for fitting and predicting using Support Vector Classifier
kfold = KFold(n_splits=5, shuffle=True, random_state=2018)
test_preds = 0
oof_preds = np.zeros([train_data.shape[0],])

for i, (train_idx,valid_idx) in enumerate(kfold.split(train_data)):
    x_train, x_valid = train_text_features_tf[train_idx,:], train_text_features_tf[valid_idx,:]
    y_train, y_valid = train_target[train_idx], train_target[valid_idx]
    classifier4 = svm.SVC()
    print('fitting.......')
    classifier4.fit(x_train,y_train)
    print('predicting......')
    print('\n')
    oof_preds[valid_idx] = classifier4.predict_proba(x_valid)[:,1]
    test_preds += 0.2*classifier4.predict_proba(test_text_features_tf)[:,1]

# training data F1 score
pred_train = (oof_preds > .25).astype(np.int)
f1_score(train_target, pred_train) 

# training data accuracy score
accuracy_score(train_target, pred_train)

# validating data F1 score
pred_valid = (oof_preds > .25).astype(np.int)
f1_score(valid_target, pred_valid) 

# validating data accuracy score
accuracy_score(valid_target, pred_valid)

# testing data F1 score
pred_test = (test_preds > .25).astype(np.int)
f1_score(test_target, pred_test)

# testing data accuracy score
accuracy_score(test_target, pred_test)

# using kfold technique for fitting and predicting using Random Forest Classifier
kfold = KFold(n_splits=5, shuffle=True, random_state=2018)
test_preds = 0
oof_preds = np.zeros([train_data.shape[0],])

for i, (train_idx,valid_idx) in enumerate(kfold.split(train_data)):
    x_train, x_valid = train_text_features_tf[train_idx,:], train_text_features_tf[valid_idx,:]
    y_train, y_valid = train_target[train_idx], train_target[valid_idx]
    classifier5 = RandomForestClassifier()
    print('fitting.......')
    classifier5.fit(x_train,y_train)
    print('predicting......')
    print('\n')
    oof_preds[valid_idx] = classifier5.predict_proba(x_valid)[:,1]
    test_preds += 0.2*classifier5.predict_proba(test_text_features_tf)[:,1]

# training data F1 score
pred_train = (oof_preds > .25).astype(np.int)
f1_score(train_target, pred_train) 

# training data accuracy score
accuracy_score(train_target, pred_train)

# validating data F1 score
pred_valid = (oof_preds > .25).astype(np.int)
f1_score(valid_target, pred_valid) 

# validating data accuracy score
accuracy_score(valid_target, pred_valid)

# testing data F1 score
pred_test = (test_preds > .25).astype(np.int)
f1_score(test_target, pred_test)

# testing data accuracy score
accuracy_score(test_target, pred_test)
