#coding:utf8

from gensim.models import Word2Vec

model = Word2Vec.load('/home/parallels/dev/ai-nlp/lesson-five/word_to_vec_model')
model_new = Word2Vec.load('/home/parallels/dev/ai-nlp/lesson-five/word_to_vec_model_new')
print(model.most_similar(u'中国'))
print(model_new.most_similar(u'中国'))

print(model.most_similar(u'性感'))
print(model_new.most_similar(u'性感'))

print(model.most_similar(u'美丽'))
print(model_new.most_similar(u'美丽'))

print(model.most_similar(u'恐怖'))
print(model_new.most_similar(u'恐怖'))

print(model.most_similar(u'艰难'))
print(model_new.most_similar(u'艰难'))

print(model.most_similar(u'色情'))
print(model_new.most_similar(u'色情'))

print(model.most_similar(u'洗衣机'))
print(model_new.most_similar(u'洗衣机'))

print(model.most_similar(u'数学'))
print(model_new.most_similar(u'数学'))

import numpy as np

words = [u'数学', u'洗衣机', '微波炉', '冰箱', '电风扇', '数学分析']
visualizeVecs=[]

for word in words:
    visualizeVecs.append(model_new[word])

from sklearn.manifold import TSNE

X_tsne = TSNE(n_components=2,random_state=33).fit_transform(visualizeVecs)
#X_tsne = TSNE().fit_transform(visualizeVecs)
print(X_tsne)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))

def display_closestwords_tsnescatterplot(model, words):
    arr = np.empty((0, 100), dtype='f')
    word_labels = []

    for word in words:
        word_labels.append(word)
        # get close words
        close_words = model.similar_by_word(word)
        print(close_words)

        # add the vector for each of the closest words to the array
        arr = np.append(arr, np.array([model[word].data]), axis=0)
        for wrd_score in close_words:
            wrd_vector = model[wrd_score[0]]
            word_labels.append(wrd_score[0])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()

display_closestwords_tsnescatterplot(model, ['冰箱', '生存', '性感', '中国', '男人', '字典'])


