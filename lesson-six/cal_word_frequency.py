#coding:utf8

sum_word_count = 0
word_freqs = {}
with open("/home/parallels/dev/ai-nlp/lesson-five/cut_words_20181223") as f:
    for line in f:
        line = line.strip()
        words = line.split()
        for word in words:
            if len(word) > 0:
                sum_word_count += 1
                if word in word_freqs:
                    word_freqs[word] += 1
                else:
                    word_freqs[word] = 1

print(sum_word_count)

for k, v in word_freqs.items():
    word_freqs[k] = float(word_freqs[k]) / sum_word_count

import pickle
pickle.dump(word_freqs, open('word_freq_dict', 'wb'))
