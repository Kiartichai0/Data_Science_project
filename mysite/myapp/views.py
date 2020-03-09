from django.shortcuts import render
from joblib import load
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
categories = ['soc.religion.christian', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# fetch_20newsgroups(subset='train', categories=categories)
# Create your views here.
def index(req):
    model = load('./myapp/static/chatgroup.model')
    result = ""
    group = ""
    # submit = 'สแดงผล'
    if req.method == 'POST':
        n = len(test.data)
        labels = model.predict(test.data)
        corrects = [ 1 for i in range(n) if test.target[i] == labels[i] ]
        score=sum(corrects)
        percentage=round((score*100/n),2)
        print('เขา POST มา')
        group = str(req.POST['group'])
        print(group)
        pred = model.predict([group])
        result = train.target_names[pred[0]]

    # def predict_category(s, train=train, model=model):
    #     pred = model.predict([s])
    #     return train.target_names[pred[0]]
    #     predict_category(group)
       
    return render(req, 'myapp/index.html',{ 
        'result': result,
        'score':score or 0,
        'total':n or 0,
        'percentage':percentage or 0

        # 'group': group, 
    })

