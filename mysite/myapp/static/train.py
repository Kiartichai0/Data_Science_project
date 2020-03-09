from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
from joblib import dump

categories = ['soc.religion.christian', 'sci.space', 'comp.graphics']
data = fetch_20newsgroups()
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)
n = len(test.data)
corrects = [ 1 for i in range(n) if test.target[i] == labels[i] ]
print(n)
print(sum(corrects))
print(sum(corrects)*100/n)
dump(model, 'chatgroup.model')