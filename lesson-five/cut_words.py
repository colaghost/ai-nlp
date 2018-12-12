#coding:utf8

import re
import jieba

sample_files = ['/home/parallels/data/wikiextractor-master/extracted/AA/zh_wiki_00', '/home/parallels/data/wikiextractor-master/extracted/AA/zh_wiki_01']
match_doc_pattern = re.compile(r'<.*>')

sample_file = open('/home/parallels/data/wikiextractor-master/extracted/AA/zh_wiki_01')

def CleanCharacters(charaters):
    return ''.join(re.findall('[\w]+', charaters))

def CutWordsAndWriteToFile(sentence, output):
    seg_list = jieba.cut(sentence, cut_all=False)
    output.write('%s' % ' '.join(seg_list))

f_output = open('out_sentences_new', 'w+')
f_output_cut = open('cut_words_new', 'w+')
p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
p2 = re.compile(r'[（\(][，；。？！\s]*[）\)]')
p3 = re.compile(r'[「『]')
p4 = re.compile(r'[」』]')
for sample_file in sample_files:
    with open(sample_file) as sample_f:
        print('process %s' % sample_file)
        sentence = ""
        for line in sample_f:
            line = match_doc_pattern.sub('', line)
            #if not line:
            #    continue
            #for c in line:
            #    if c in u'，。、？！：':
            #        sentence = CleanCharacters(sentence)
            #        if sentence:
            #            CutWordsAndWriteToFile(sentence, f_output_cut)
            #            print(sentence)
            #            f_output.write('%s\n' % sentence)
            #        sentence = ""
            #    else:
            #        sentence += c
            line = p1.sub(r'\2', line)
            line = p2.sub(r'', line)
            line = p3.sub(r'“', line)
            line = p4.sub(r'”', line)
            f_output.write(line)
            CutWordsAndWriteToFile(line, f_output_cut)
        #if not sentence:
        #    CutWordsAndWriteToFile(sentence, f_output_cut)
        #    f_output.write('%s\n' % sentence)
        #    sentence = ""

f_output.close()
