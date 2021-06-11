#Hyperlink of Youtube Video Tutorial
#https://www.youtube.com/watch?v=M9Itm95JzL0

import random

class Sentiment:
    NEGATIVE = 'NEGATIVE'
    NEUTRAL = 'NEUTRAL'
    POSITIVE = 'POSITIVE'

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else: #Score = 4 or 5
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def evenly_distribute(self):
        negative = list(filter(lambda x:x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x:x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)


import json


file_name = "./data_csv/Books_small_10000.json"

reviews = []

with open (file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

print(reviews[5].text)
print(reviews[5].score)
print(reviews[5].sentiment)

#know the number of reviews
print(len(reviews))

#CONVERT TEXT INTO QUANTITATIVE VECTORS - MUST for machine learning on text data

from sklearn.model_selection import train_test_split
#0.33 = 33% of the reviews will be used in test training
training, test = train_test_split(reviews, test_size=0.33, random_state=42)
#below to print the trainins set and the test set
#this is training: print(len(training))  -- this is test: #print(len(test))

train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

train_x = [x.text for x in training]
train_y = [x.sentiment for x in training]

test_x = [x.text for x in test]
test_y = [x.sentiment for x in test]

#Bags of words vectorization
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

#fit_transform = fit and transform
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

print(train_x[0])
print(train_x_vectors[0].toarray())

# CLASSIFICATION
#Liner SVM
from sklearn import svm

clf_svm = svm.SVC(kernel = 'linear')
clf_svm.fit(train_x_vectors, train_y)

print(test_x[0])

print(clf_svm.predict(test_x_vectors[0]))

###Decision Three
from sklearn.tree import DecisionTreeClassifier

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

print(clf_dec.predict(test_x_vectors[0]))

###Naive Bayes
from sklearn.naive_bayes import GaussianNB

clf_gnb = DecisionTreeClassifier()
clf_gnb.fit(train_x_vectors, train_y)

print(clf_gnb.predict(test_x_vectors[0]))

###Logistic Regresion
from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)
print(clf_log.predict(test_x_vectors[0]))

###EVALUATION
#Mean accuracy
print(clf_svm.score(test_x_vectors, test_y))
print(clf_dec.score(test_x_vectors, test_y))
print(clf_gnb.score(test_x_vectors, test_y))
print(clf_log.score(test_x_vectors, test_y))

#F1 Scores
from sklearn.metrics import f1_score

print(f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_dec.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_gnb.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_log.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE]))

print(train_x[0])
print(train_y[0:5])
print(train_y.count(Sentiment.POSITIVE))

