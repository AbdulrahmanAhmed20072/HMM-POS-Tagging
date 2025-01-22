from utils_pos import get_word_tag, preprocess
import pandas as pd
from collections import defaultdict
import numpy as np

# load the training corpus
with open('WSJ_02-21.pos','r') as f:
    training_corpus = f.readlines()

print(training_corpus[:5])

# read the Hidden Markov Models vocabulary

with open('hmm_vocab.txt','r') as f:
    voc_l = f.read().split('\n')

print(voc_l[-10:])

vocab = {}
for i, word in enumerate(sorted(voc_l)):
    vocab[word] = i

# load the test corpus
with open("WSJ_24.pos", 'r') as f:
    y = f.readlines()
y[:5]

_, preprocessed = preprocess(vocab, "test.words")
print(len(preprocessed))

preprocessed[:5]

def create_dictionaries(training_corpus, vocab):

    # used between pos and pos
    transition_count = defaultdict(int)
    # used between pos and word
    emission_count = defaultdict(int)
    tag_count = defaultdict(int)

    prev_tag = '--s--'

    for i, line in enumerate(training_corpus):

        word, tag = get_word_tag(line, vocab)

        transition_count[(prev_tag, tag)] += 1
        emission_count[(tag, word)] += 1
        tag_count[tag] += 1

        prev_tag = tag

    return transition_count, emission_count, tag_count

transition_count, emission_count, tag_count = create_dictionaries(training_corpus, vocab)

len(transition_count), len(emission_count), len(tag_count)

print("transition_count:")
print({k : v for k,v in list(transition_count.items())[:5]})
print()
print("emission_count:")
print({k : v for k,v in list(emission_count.items())[:5]})
print()
print("tag_count:")
print({k : v for k,v in list(tag_count.items())[:5]})

states = sorted(tag_count.keys())
print(states[-5:])

def predict_pos(preprocessed, y, emission_count, vocab,states):

    words_dict, res = defaultdict(int), []

    for word in preprocessed:
        for tag in states:

            if emission_count.get((tag, word)):
                if word not in words_dict:

                    words_dict[word] = (tag,emission_count[(tag, word)])

                else:
                    if words_dict[word][1] < emission_count[(tag, word)]:
                        words_dict[word] = (tag, emission_count[(tag, word)])

    for i in y:

        word,tag = get_word_tag(i,words_dict)
        if word in words_dict:
            res.append(tag == words_dict[word][0])

    return sum(res) / len(y)

predict_pos(preprocessed,y,emission_count,vocab, states)

def predict_pos2(preprocessed, y, emission_count, vocab,states):

    words_dict, res = defaultdict(int), []

    for word, y_tup in zip(preprocessed,y):
        for tag in states:

            if emission_count.get((tag, word)):
                if word not in words_dict:

                    words_dict[word] = (tag,emission_count[(tag, word)])

                else:
                    if words_dict[word][1] < emission_count[(tag, word)]:
                        words_dict[word] = (tag, emission_count[(tag, word)])

        word,tag = get_word_tag(y_tup, words_dict)

        if word in words_dict:
            res.append(tag == words_dict[word][0])

    return sum(res) / len(y)

predict_pos2(preprocessed,y,emission_count,vocab, states)

def create_transition_matrix(alpha, states, transition_count):

    # transition_matrix contains the proba between pos and pos
    num_tags = len(states)
    transition_matrix = np.zeros((num_tags, num_tags))

    for i, tag_row in enumerate(states):
        for j, tag_col in enumerate(states):

            transition_matrix[i,j] = (transition_count[(tag_row, tag_col)] + alpha) / (tag_count[tag_row] + alpha * num_tags)

    return transition_matrix

# A is the transition matrix
A = create_transition_matrix(.001,states,transition_count)
A_df = pd.DataFrame(A, index = states, columns = states)

A_df.iloc[30:35,30:35]

def create_emission_matrix(alpha, emission_count, states, vocab):

    num_words = len(vocab)
    num_tags = len(states)
    B = np.zeros((num_tags , num_words))

    for i in range(num_tags):
        for j in range(num_words):

            pair = (states[i], vocab[j])
            B[i, j] = (emission_count[pair] + alpha) / (tag_count[states[i]] + alpha * num_tags)

    return B

B = create_emission_matrix(0.001, emission_count, states, list(vocab))
B_df = pd.DataFrame(B, index=states , columns = list(vocab))

B_df.iloc[35:40,35:40]

def initialize(states, tag_count, A, B, corpus, vocab):

    # this func used to fill the first column of best_probs

    num_tags = len(states)
    num_words = len(preprocessed)
    best_probs = np.zeros((num_tags, num_words))
    best_paths = np.zeros((num_tags, num_words), dtype= int)
    s_idx = states.index('--s--')

    for i in range(num_tags):

        word_idx = corpus.index(corpus[0])

        pos_pos = A[(s_idx, i)]
        pos_word = B[(i, word_idx)]

        best_probs[i,0] = np.log(pos_pos) + np.log(pos_word)

    return best_probs, best_paths

best_probs, best_paths = initialize(states, tag_count, A, B, preprocessed, vocab)

# num_tags * num_words
best_probs.shape

def viterbi_forward(A,B, best_probs, best_paths, vocab, corpus):

    # this func used to fill all best_probs

    num_tags = A.shape[0] # tags * words

    # for each word, we get the proba for the 46 tags
    for word in range(1, len(corpus)):
        for tag in range(num_tags):

            # the previous word, get max proba in column
            prev_max_proba = best_probs[:, word-1].max()
            prev_idx = best_probs[:, word-1].argmax()

            transition_proba = A[prev_idx, tag]
            emission_proba = B[tag, vocab[preprocessed[word]]]

            best_probs[tag, word] = prev_max_proba + np.log(transition_proba) + np.log(emission_proba)

        cur_idx = best_probs[:, word].argmax()
        best_paths[cur_idx, word] = prev_idx

    return best_probs, best_paths

best_probs_f, best_paths_f = viterbi_forward(A,B, best_probs, best_paths, vocab, preprocessed)

print(best_paths_f[:10])

print(best_probs_f[:,4])

# Test this function
print(f"best_probs[0,1]: {best_probs_f[0,1]:.4f}")
print(f"best_probs[0,4]: {best_probs_f[0,4]:.4f}")

def viterbi_backward(best_probs, best_paths, corpus, states):

    # viterbi_backward used for prediction

    res = [None] * best_paths.shape[1]
    res[-1] = states[best_probs[:,-1].argmax()]

    for i in range(best_paths.shape[1]-2,-1,-1):

        res[i] = states[best_paths[:, i].argmax()]

    return res

pred = viterbi_backward(best_probs_f, best_paths_f , preprocessed, states)

for i in range(10):
    print(pred[i],y[i].split()[1])

m=len(pred)
print('The prediction for pred[-7:m-1] is: \n', preprocessed[-7:m-1], "\n", pred[-7:m-1], "\n")

print('The third word is:', preprocessed[3])
print('Your prediction is:', pred[3])
print('Your corresponding label y is: ', y[3])

def compute_accuracy(pred, y):

    num_correct = 0
    total = 0
    for prediction, y in zip(pred, y):
        # Split the label into the word and the POS tag
        word_tag_tuple = y.split()

        # Check that word and tag are not none
        if len(word_tag_tuple)==2:

            word, tag = word_tag_tuple

            if prediction == tag:
                num_correct += 1

            total += 1

    return (num_correct/total)

print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")

