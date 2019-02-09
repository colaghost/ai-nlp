import pandas as pd
import xgboost as xgb
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import decomposition, ensemble


content = pd.read_csv('~/dev/ai-nlp/lesson-six/sqlResult_1558435.csv', encoding='gb18030')
content = content.fillna('')

def read_stop_words():
    stop_words = []
    with open('/home/parallels/dev/ai-nlp/lesson-eleven/stop_words_cn.txt', 'r') as f:
        for word in f:
            stop_words.append(word.strip())
    return stop_words

def readtrain():
    content_train = []
    opinion_train = []
    for idx in range(len(content['content'])):
        main_content = content['content'][idx]
        source = content['source'][idx]
        if main_content.find('新华社') != -1 and source.find('新华') == 0:
            continue
        content_train.append(main_content)
        opinion_train.append(1 if source.find('新华') == 0 else 0)
    choose_len = int(len(content_train))
    return [content_train[:choose_len], opinion_train[:choose_len]]

def segment_word(content):
    c = []
    for sentence in content:
        words = list(jieba.cut(sentence))
        c.append(" ".join(words))
    return c

stop_words = read_stop_words()
train = readtrain()
content = segment_word(train[0])
opinion = np.array(train[1])
vect = TfidfVectorizer(min_df=2, analyzer='word', token_pattern=r'\w{1,}', stop_words=stop_words)
xvec = vect.fit_transform(content)

train_content, test_content, train_opinion, test_opinion = train_test_split(xvec, opinion, train_size=0.7, random_state=1)

dtrain = xgb.DMatrix(train_content, label=train_opinion)
dtest = xgb.DMatrix(test_content, label=test_opinion)
param = {'max_depth':6, 'eta':0.1, 'eval_metric':'auc', 'silent':0, 'objective':'binary:logistic'}  # 参数
evallist  = [(dtrain,'train'), (dtest,'test')]
num_round = 50
bst = xgb.train(param, dtrain, num_round, evallist)
preds = bst.predict(dtest)

#classifier = naive_bayes.MultinomialNB()
#classifier = linear_model.LogisticRegression()
#classifier = svm.SVC()
#classifier = ensemble.RandomForestClassifier()
#classifier.fit(train_content, train_opinion)
#preds = classifier.predict(test_content)
y_pred = (preds >= 0.5) * 1

from sklearn import metrics

print ('AUC: %.4f' % metrics.roc_auc_score(test_opinion,preds))
print ('ACC: %.4f' % metrics.accuracy_score(test_opinion,y_pred))
print ('Recall: %.4f' % metrics.recall_score(test_opinion,y_pred))
print ('Precesion: %.4f' % metrics.precision_score(test_opinion,y_pred))

#AUC: 0.9871
#ACC: 0.9899
#Recall: 0.9436
#Precesion: 0.9913