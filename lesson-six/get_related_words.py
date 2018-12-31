#coding:utf8

from gensim.models import Word2Vec

model = Word2Vec.load('/home/parallels/dev/ai-nlp/lesson-five/word_to_vec_model')
model_new = Word2Vec.load('/home/parallels/dev/ai-nlp/lesson-five/word_to_vec_model_20181223')

from collections import defaultdict

def cal_score(layer, similarity):
    return similarity / layer

def get_related_words(initial_words, model):
    max_size = 500
    seen = defaultdict(float)
    unseen = [[initial_words, 1.0, 1]]

    while unseen and len(seen) < max_size:
        if len(seen) % 100 == 0:
            print('seen length: {}'.format(len(seen)))
        node = unseen.pop(0)
        word = node[0]
        similar = node[1]
        layer = node[2]

        #new_expanding = [w for w, s in model.most_similar(node, topn=20)]
        new_expanding = []

        for w, s in model.most_similar(word, topn=20):
            if w != word:
                new_expanding.append([w, s, layer + 1])

        unseen += new_expanding

        seen[word] += cal_score(layer, similar)
        # if we need more sophsiticated, we need change the value as the function(layer, similarity)

    return seen

related_words = get_related_words('说', model_new)
print(sorted(related_words.items(), key=lambda x:x[1], reverse=True)[:100])