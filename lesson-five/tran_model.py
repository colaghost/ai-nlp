#coding: utf8
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sentences_file = '/home/parallels/dev/ai-nlp/lesson-five/cut_words_new'
output_model_file = '/home/parallels/dev/ai-nlp/lesson-five/word_to_vec_model_new'

#model = Word2Vec(LineSentence(sentences_file), window=5, workers=multiprocessing.cpu_count())
model = Word2Vec(LineSentence(sentences_file), size=200, window=5, min_count=5, workers=4)
model.save(output_model_file)