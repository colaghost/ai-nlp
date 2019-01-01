from gensim.models import Word2Vec
import pickle
import numpy as np
from typing import List
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pyltp import Parser
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import SentenceSplitter
import os
import jieba

model = Word2Vec.load('/home/parallels/dev/ai-nlp/lesson-five/word_to_vec_model_20181223')
par_model_path = os.path.join('/home/parallels/dev/model/ltp_data_v3.4.0', 'parser.model')
parser = Parser()
parser.load(par_model_path)
cws_model_path = os.path.join('/home/parallels/dev/model/ltp_data_v3.4.0', 'cws.model')
segmentor = Segmentor()
segmentor.load(cws_model_path)
pos_model_path = os.path.join('/home/parallels/dev/model/ltp_data_v3.4.0', 'pos.model')
postagger = Postagger()
postagger.load(pos_model_path)
ner_model_path = os.path.join('/home/parallels/dev/model/ltp_data_v3.4.0', 'ner.model')
recognizer = NamedEntityRecognizer()
recognizer.load(ner_model_path)


class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector


class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    def len(self) -> int:
        return len(self.word_list)


def get_word_frequency(word_text, looktable):
    if word_text in looktable:
        return looktable[word_text]
    else:
        return 1.0

def raw_sentence_to_vec(sentence_list: List[Sentence], embedding_size, looktable, a=1e-3):
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = sentence.len()
        for word in sentence.word_list:
            a_value = a / (a + get_word_frequency(word.text, looktable))  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word.vector))  # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences

    # calculate PCA of this sentence set
    #pca = PCA(n_components=embedding_size)
    pca = PCA()
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))

    return sentence_vecs


freq_dict_path = '/home/parallels/dev/ai-nlp/lesson-six/word_freq_dict'
embedding_size = 200
freq_dict = {}
with open(freq_dict_path, 'rb') as f:
    freq_dict = pickle.load(f)

def sentence_to_vector(sentence):
    s = []
    for word in jieba.cut(sentence):
        try:
            vec = model[word]
        except:
            vec = np.zeros(embedding_size)
        s.append(Word(word, vec))
    ss = Sentence(s)
    vectors = raw_sentence_to_vec([ss], embedding_size=embedding_size, looktable=freq_dict)
    return vectors[0]

#sentence1 = "我是一个中国人"
#sentence2 = "我是一个中华人民共和国人"
#
#s1 = []
#s2 = []
#all_senteces = []
#
#for word in jieba.cut(sentence1):
#    try:
#        vec = model[word]
#    except:
#        vec = np.zeros(embedding_size)
#    s1.append(Word(word, vec))
#
#for word in jieba.cut(sentence2):
#    try:
#        vec = model[word]
#    except:
#        vec = np.zeros(embedding_size)
#    s2.append(Word(word, vec))
#
#ss1 = Sentence(s1)
#ss2 = Sentence(s2)
#all_senteces.append(ss1)
#all_senteces.append(ss2)
#
#sentence_vectors = sentence_to_vec(all_senteces, embedding_size, looktable=freq_dict)
#len_sentences = len(sentence_vectors)
#for i in range(len_sentences):
#    if i % 2 == 0:
#        sim = cosine_similarity([sentence_vectors[i]], [sentence_vectors[i + 1]])
#        print(sim)

def get_full_entity(words, entity_idx, postags, nertags, relation, heads, rely_id):
    if nertags[entity_idx] != 'O':
        if nertags[entity_idx].find('-') == -1:
            return words[entity_idx]
        first_half = []
        second_half = [words[entity_idx]]
        idx = entity_idx - 1
        if nertags[entity_idx][0:2] in ['I-', 'E-']:
            while idx >= 0:
                first_half.append(words[idx])
                if nertags[idx][0:2] == 'B-':
                    break
                idx -= 1
        idx = entity_idx + 1
        if nertags[entity_idx][0:2] in ['B-', 'I-']:
            while idx < len(words):
                second_half.append(words[idx])
                if nertags[idx][0:2] == 'E-':
                    break
                idx += 1
        return ''.join(first_half[::-1]) + ''.join(second_half)
    entity_words = [words[entity_idx]]
    curr_idx = entity_idx - 1
    while curr_idx >= 0 and relation[curr_idx] == 'ATT' and rely_id[curr_idx] - 1 == curr_idx + 1:
        entity_words.append(words[curr_idx])
        curr_idx -= 1
    return ''.join(entity_words[::-1])

def locate_speech_entity_for_sentence(sentence, postagger, parser):
    talk_words = set('透露 留言 说 怒道 质问 曾言 喊道 写道 笑答 地问 脱口而出 答 大喊 哭道 直言 告诫 表示 痛骂 破口大骂 大叫 著说 时说 坦言 怒斥 辩称 问说 责备 询问 唱道 中说 宣称 公曰 追问 还称 有言 喃喃自语 责骂 斥责 答说 慨叹 地说 谈到 问过 臣闻 坚称 说出 坦承 评说 韦昭注 叮嘱 自言 认为 信中称 奏道 诗云 惊呼 不禁 提到 宣宗称 文说 评曰 透露 所说 请问 自谓 怒骂 谚云 大呼 否认 骂 问起 指责 痛斥 惊道 劝道 指称 称 喝道 中称 言道 公说 相信 如是说 强调 说明 注云 重申 表明 辱骂 介绍'.split())
    words = list(jieba.cut(sentence))
    postags = list(postagger.postag(words))
    arcs = parser.parse(words, postags)  # 句法分析
    nertags = list(recognizer.recognize(words, postags))

    rely_id = [arc.head for arc in arcs]    # 提取依存父节点id
    relation = [arc.relation for arc in arcs]   # 提取依存关系
    heads = ['Root' if id == 0 else words[id-1] for id in rely_id]  # 匹配依存父节点词语

    for i in range(len(words)):
        if relation[i] == 'SBV' and (heads[i] in talk_words):
            if postags[i][0] != 'n':
                break
            entity = get_full_entity(words, i, postags, nertags, relation, heads, rely_id)
            talk_word_id = rely_id[i]-1
            talk_subject_id_start_idx = talk_word_id
            last_talk_word_id = talk_word_id
            while last_talk_word_id + 1 < len(words):
                if rely_id[last_talk_word_id + 1] - 1 == last_talk_word_id and relation[last_talk_word_id + 1] in set(['COO', 'RAD', 'LAD']) :
                    last_talk_word_id += 1
                elif relation[last_talk_word_id +1] == 'WP' and words[last_talk_word_id + 1] != '“':#in set(['：', ',', '，', ',', ':']):
                    last_talk_word_id += 1
                else:
                    break
            if last_talk_word_id + 1 < len(words):
                if words[last_talk_word_id + 1] == '“' and words[-1] != '”':
                    quotation_mark_pos = -1
                    try:
                        quotation_mark_pos = words.index('”', last_talk_word_id + 1)
                    except:
                        pass
                    if quotation_mark_pos != -1 and words[quotation_mark_pos + 1] in set([',', '，']):
                        return (entity, ''.join(words[last_talk_word_id + 1:quotation_mark_pos + 1]), True)
                return (entity, ''.join(words[last_talk_word_id+1:]), words[-1] == '”')
            else:
                idx = i - 1
                while idx >= 0:
                    if postags[idx] == 'wp':
                        if words[idx] == '”':
                            idx += 1
                            break
                    idx -= 1
                if idx > 0:
                    return (entity, ''.join(words[0:idx]), True)
    return None

def split_sentences(content):
    sentences = SentenceSplitter.split(content)
    return list(sentences)

content = pd.read_csv('~/dev/ai-nlp/lesson-six/sqlResult_1558435.csv', encoding='gb18030')
content = content.fillna('')
all_news_content = content['content']
i = 0
for news in all_news_content:
    sentences = split_sentences(news)
    idx = 0
    while idx < len(sentences):
        sentence = sentences[idx]
        idx += 1
        result = locate_speech_entity_for_sentence(sentence, postagger, parser)
        if not result:
            continue
        i += 1
        #print('{} | {}'.format(result[0], result[1]))
        speechs = [result[1]]
        speech = result[1]
        #说明发表的言论可能跨句了
        if not result[2]:
            #这些情况下的言论认为是被“”包住的
            if speech[0] == '“':
                quotation_mark_pos = speech.find('”')
                if quotation_mark_pos == -1:
                    while idx < len(sentences):
                        sentence = sentences[idx]
                        idx += 1
                        pos = sentence.find('”')
                        speechs.append(sentence)
                        if pos != -1:
                            break
            #这种情况下认为句子与句子之间相似度接近则认为都是发表的言论
            else:
                return_speech_vec = sentence_to_vector(result[1])
                while idx < len(sentences):
                    sentence = sentences[idx]
                    idx += 1
                    if len(sentence.strip()) == 0:
                        continue
                    sentence_vec = sentence_to_vector(sentence)
                    sim = cosine_similarity([return_speech_vec], [sentence_vec])
                    if sim >= 0.63 and not locate_speech_entity_for_sentence(sentence, postagger, parser):
                        speechs.append(sentence)
                    else:
                        idx -= 1
                        break
        print('{} | {}'.format(result[0], ''.join(speechs)))
