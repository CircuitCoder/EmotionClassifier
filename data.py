import re
from random import shuffle
import math

TRAIN_INPUT = '/root/data/sinanews.train'
TEST_INPUT = '/root/data/sinanews.test'
EMBED_INPUT = '/root/learn/embed.txt'
TF_IDF_THRESHOLD = 0
MIN_EMOJIS = 0

with open('./stopwords.txt') as f:
    stopwords = set(f.readlines())

vocab = set()
word_to_idx = dict()
embeds = []
counter = 0

with open(EMBED_INPUT) as f:
    for line in f.readlines()[1:]:
        segs = line.strip().split(' ')
        word = segs[0]
        try:
            data = [float(s) for s in segs[1:]]
        except:
            print(segs[1:])
            raise 1
        vocab.add(word)
        word_to_idx[word] = counter
        counter += 1
        embeds += [data]
        if counter % 10000 == 0:
            print('Reading embedding... {}'.format(counter))

with open(TRAIN_INPUT) as f:
    lines = f.readlines()
with open(TEST_INPUT) as f:
    test_lines = f.readlines()

longest = -1

def ignore_word(w):
    if w in stopwords:
        return True
    if any(char.isdigit() for char in w):
        return True
    return w not in vocab

def process(l, check_emoji = False):
    [raw_emojis, raw_words] = re.compile("Total:\d+").split(l)[1].strip().split('\t')
    num_emojis = [int(e.split(':')[1]) for e in raw_emojis.strip().split(' ')]
    # Normalize
    tot_emojis = sum(num_emojis)
    emojis = [e / tot_emojis for e in num_emojis]
    if check_emoji and tot_emojis < MIN_EMOJIS:
        return None

    words = [w for w in raw_words.strip().split(' ') if not ignore_word(w)]

    return [emojis, words]

raw_data = [process(l, True) for l in lines]
raw_data = [l for l in raw_data if l is not None]

def idx(l):
    [emojis, all_words] = l
    
    words = [ word_to_idx[word] for word in all_words ]

    global longest
    longest = max(longest, len(words))

    return [ emojis, words ]

data = [idx(l) for l in raw_data]

raw_test_data = [process(l) for l in test_lines]
test_data = [idx(l) for l in raw_test_data]

test_data.sort(key = lambda s: len(s[1]), reverse=True)
print(len(data))

print("Input complete")
print("Longest sentence: " + str(longest), flush=True)

VOCAB_SIZE = len(vocab)
