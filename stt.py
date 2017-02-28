import sys
import random
from math import floor
from collections import defaultdict

import utils

def prepare():
    data = load_dictionary()
    labels = load_labels()
    char_count = defaultdict(int)
    word_count = defaultdict(int)
    label_file = open(utils.LABELS_FILENAME, "wt", encoding="utf8")
    sentences_file = open(utils.SENTENCES_FILENAME, "wt", encoding="utf8")
    splitted_sentences_file = open(utils.SPLITTED_SENTENCES_FILENAME, "wt", encoding="utf8")
    for idx, sentence in enumerate(data.values()):
        if idx%1000 == 0:
            sys.stdout.write("\r{} / {}".format(idx, len(data)))
        for char in sentence:
            char_count[char] += 1
        words = sentence.split(" ")
        for word in words:
            word_count[word] += 1
        label_file.write(str(labels[idx])+"\n")
        sentences_file.write(sentence+" "+utils.EOS_WORD+"\n")
        splitted_sentences_file.write(sentence+" "+utils.EOS_WORD+"\n")

    indexes = list(range(len(labels)))
    random.shuffle(indexes)

    with open(utils.INDEXES_FILENAME, "wt", encoding="utf8") as f:
        for index in indexes:
            f.write(str(index)+"\n")

    word_corpus = list(word_count.keys())
    word_corpus.sort()
    char_corpus = list(char_count.keys())
    char_corpus.sort()
    char_corpus_file = open(utils.CHAR_CORPUS_FILENAME, "wt")
    char_counts_file = open(utils.CHAR_COUNTS_FILENAME, "wt")
    word_corpus_file = open(utils.WORD_CORPUS_FILENAME, "wt")
    word_counts_file = open(utils.WORD_COUNTS_FILENAME, "wt")
    for char in char_corpus:
        char_corpus_file.write(char)
        char_counts_file.write(str(char_count[char])+"\n")
    for word in word_corpus:
        word_corpus_file.write(word+"\n")
        word_counts_file.write(str(word_count[word])+"\n")
    char_corpus_file.close()
    char_counts_file.close()
    word_corpus_file.close()
    word_counts_file.close()


def load_dictionary(split=False):
    result = {}
    sys.stdout.write('Loading dictionary\n')
    with open("STT/dictionary.txt", "rt") as f:
        for row in f:
            data = row.split("|")
            phrase = data[0]
            phrase_id = int(data[1])
            if split:
                phrase = phrase.split(" ")
            result[phrase_id] = phrase
    return result

def load_labels():
    result = []
    with open("STT/sentiment_labels.txt", "rt") as f:
        for idx, row in enumerate(f):
            if idx == 0:
                continue
            label = float(row.split("|")[1])
            label = floor(label*5)
            if label == 5:
                label = 4
            result.append(label)
    return result

if __name__ == "__main__":
    prepare()