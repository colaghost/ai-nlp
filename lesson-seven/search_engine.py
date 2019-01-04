import jieba
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import reduce
from operator import and_
from scipy.spatial.distance import cosine

file_dir = '/home/parallels/data/bbs_doc'

class SearchEngine:
    def __init__(self, file_dir):
        self.corpus = []
        if os.path.exists(os.path.join(file_dir, 'corpus')):
            with open(os.path.join(file_dir, 'corpus'), "rb") as f:
                self.corpus =  pickle.load(f)
        else:
            idx = 0
            for file_name in os.listdir(file_dir):
                idx += 1
                if idx > 3000:
                    break
                path = os.path.join(file_dir, file_name)
                with open(path) as f:
                    content = f.read()
                    words = list(jieba.cut(content))
                    self.corpus.append(' '.join(words))
            pickle.dump(self.corpus, open(os.path.join(file_dir, 'corpus'), 'wb'))
        self.vectorizer = TfidfVectorizer()
        self.tfidf = self.vectorizer.fit_transform(self.corpus)
        transposed_tfidf = self.tfidf.transpose()
        print(transposed_tfidf.shape)
        self.transposed_tfidf_array = transposed_tfidf.toarray()

    def get_word_id(self, word):
        return self.vectorizer.vocabulary_.get(word, None)

    def get_content_ids(self, query_words):
        ids = []
        for word in query_words:
            id = self.get_word_id(word)
            if id:
                ids.append(id)
        return ids

    def search(self, query_string):
        query_string = query_string.strip()
        query_words = jieba.cut(query_string)
        content_ids = self.get_content_ids(query_words)
        if len(content_ids) == 0:
            return None
        contents = []
        for content_id in content_ids:
            contents.append(set(np.where(self.transposed_tfidf_array[content_id])[0]))
        merged_contents = reduce(and_, contents)
        if len(merged_contents) == 0:
            return None
        vector_with_id = [(self.tfidf[i], i) for i in merged_contents]
        query_vec = self.vectorizer.transform([' '.join(query_words)]).toarray()[0]
        sorted_vector_with_id = sorted(vector_with_id, key=lambda k: cosine(k[0].toarray(), query_vec))
        sorted_ids = [i for v, i in sorted_vector_with_id]
        print(sorted_ids)
        contents = []
        for id in sorted_ids:
            contents.append(self.corpus[id])
        return contents[:10]

search_engine = SearchEngine(file_dir)
contents = search_engine.search('上海 制作')
if contents:
    for content in contents:
        print(content)
