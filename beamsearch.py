import numpy as np
import math
import heapq

# Source: https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

def beam_search_decoder(data, k):
    """
    data: (n, m) where n is number of words in sequence.
        and m is number of classes (words in target vocab).
    k: beam search parameter
    """
    sequences = [[[], 0.0]]
    # walk over each step in sequence
    for row in data: # ----> n
        all_candidates = []
        # find the indexes of k largest probabilities in the row
        k_largest = heapq.nlargest(k, range(len(row)), row.take) # -----> m
        # expand each current candidate
        for seq, score in sequences: # ----> k
            for j in k_largest: # -----> k
                s = score - math.log(row[j])
                candidate = [seq + [j], s]
                all_candidates.append(candidate)
        # sort all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1]) # -----> k log k
        # select best k
        sequences = ordered[:k]
    return sequences


if __name__ == '__main__':

    # define a sequence of 10 words over a vocab of 5 words
    data = [[0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1]]
    
    data = np.array(data)
    print(data.shape)
    # decode sequence
    result = beam_search_decoder(data, 3)
    # print result
    for seq in result:
        print(seq)