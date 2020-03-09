from django.shortcuts import render
from joblib import load
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
categories = ['soc.religion.christian', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)

# fetch_20newsgroups(subset='train', categories=categories)
# Create your views here.
def index(req):
    model = load('./myapp/static/chatgroup.model')
    result = ""
    group = ""
    # submit = 'สแดงผล'
    if req.method == 'POST':
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
        # 'group': group, 
    })

