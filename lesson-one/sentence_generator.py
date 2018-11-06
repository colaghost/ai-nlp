#coding:utf8
import random

simple_grammar = """
sentence => noun_phrase verb_phrase 
noun_phrase => Article Adjs noun
Adjs => null | Adj Adjs
verb_phrase => verb noun_phrase
Article => 一个 | 这个 | Adj
noun => 女人| 篮球|桌子|小猫
verb => 看着 | 坐在| 听着 | 看见
Adj => 蓝色的| 好看的 | 小小的 """

def GenGrammarSymbolTable(grammar):
    symbol_table = {}
    lines = grammar.split("\n")
    for line in lines:
        if not line:
            continue
        fields = line.split('=>')
        if len(fields) != 2:
            continue
        choices = fields[1].strip().split('|')
        symbol_table[fields[0].strip()] = [choice.strip().split() for choice in choices]
    return symbol_table

def GenSentence(symbol_table, curr_symbol, words):
    if curr_symbol in symbol_table:
        choice = random.choice(symbol_table[curr_symbol])
        for symbol in choice:
            GenSentence(symbol_table, symbol, words)
    else:
        if curr_symbol == 'null':
            return
        else:
            words.append(curr_symbol)

symbol_table = GenGrammarSymbolTable(simple_grammar)
words = []
GenSentence(symbol_table, 'sentence', words)
print(''.join(words))