#coding:utf8

import pandas as pd
import re
from collections import Counter
import jieba
import os

def CleanCharacters(charaters):
    return ''.join(re.findall('[\w]+', charaters))


#sample_file = open('/home/parallels/data/80k_articles.txt')
match_doc_pattern = re.compile(r'<.*>')
sample_file = open('/home/parallels/data/wikiextractor-master/extracted/AA/zh_wiki_00')
clean_characters = ""
while 1:
    line = sample_file.readline()
    if not line:
        break
    line = match_doc_pattern.sub('', line)
    line = CleanCharacters(line)
    clean_characters += line

sample_file.close()

clean_characters_count = Counter(clean_characters)

def GetProbabilityTemplate(counter):
    character_count = sum(counter.values())
    element_freq_map = {}
    for (e, c) in counter.most_common():
        element_freq_map.setdefault(c, 0)
        element_freq_map[c] += 1
    total = 0
    items = element_freq_map.items()
    items = sorted(items)[:11]
    lt11_dict = {freq:count for freq, count in items}
    print(lt11_dict)
    lt11_probability = {}
    total_probability = 0
    total_recal_probability = 0
    for freq, count in items:
        if freq < 11:
            lt11_probability.setdefault(freq, 0.0)
            dr = ((freq + 1) * (lt11_dict[freq + 1] + 1)) / (lt11_dict[freq] + 1)
            lt11_probability[freq] = dr / character_count
            total_recal_probability += lt11_probability[freq] * lt11_dict[freq]
            total_probability += freq / character_count * lt11_dict[freq]
            print("{} {}".format(freq, freq/character_count))
    lt11_probability[0] = 1 / character_count
    print(lt11_probability)
    print("{} {}".format(total_recal_probability, total_probability))
    def GetProbability(character):
        freq = counter[character] if character in counter else 0
        if freq <= 10:
            return lt11_probability[freq]
        return counter[character] / character_count
    return GetProbability

GetProbabilityFromCleanCountersFunc = GetProbabilityTemplate(clean_characters_count)

def GetProbabilityOfSentence(sentence):
    clean_sentence = CleanCharacters(sentence)
    probability = 1.0
    for character in clean_sentence:
        temp_probability = GetProbabilityFromCleanCountersFunc(character)
        probability *= temp_probability
    return probability

pair = """前天晚上吃晚饭的时候 前天晚上吃早饭的时候""".split()

pair2 = """正是一个好看的小猫 真是一个好看的小猫""".split()

pair3 = """我无言以对，简直 我简直无言以对""".split()

pairs = [pair, pair2, pair3]
for pair in pairs:
    print("{} {}".format(GetProbabilityOfSentence(pair[0]), GetProbabilityOfSentence(pair[1])))

#word_list = []
#
#if not os.path.exists('cut_word_list'):
#    cut_word_seg = jieba.cut(clean_characters, cut_all=False)
#    for word in cut_word_seg:
#        word_list.append(word)
#    with open('cut_word_list', "w+") as cut_word_list_file:
#        for word in word_list:
#            cut_word_list_file.write('%s\n' % word)
#else:
#    with open('cut_word_list') as cut_word_list_file:
#        for word in cut_word_list_file:
#            word_list.append(word.strip())
#
#one_gram_counter = Counter(word_list[i] for i in range(len(word_list) - 1))
#GetOneGramProbabilityFromCleanCountersFunc = GetProbabilityTemplate(one_gram_counter)

gram_count = 2
two_gram_counter = Counter(clean_characters[i:i+gram_count] for i in range(len(clean_characters) - gram_count))
GetTwoGramProbabilityFromCleanCountersFunc = GetProbabilityTemplate(two_gram_counter)

def GetTwoGramProbability(character, prev):
    return GetTwoGramProbabilityFromCleanCountersFunc(prev+character)

def GetTwoParamProbabilityForSentence(sentence):
    probability = 1.0
    for idx, character in enumerate(sentence):
        if idx == 0:
            probability *= GetProbabilityFromCleanCountersFunc(character)
        else:
            probability *= GetTwoGramProbability(character, sentence[idx-1])
    return probability

for pair in pairs:
    print("{} {}".format(GetTwoParamProbabilityForSentence(pair[0]), GetTwoParamProbabilityForSentence(pair[1])))
