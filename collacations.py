import sys
import math
from collections import Counter, defaultdict

def read_corpus(filename):
    with open(filename, 'r') as file:
        text = file.read()

    words = [word for word in text.split() if not word in ',.!?:;-`"\'’()/']
    return words

def calculate_counts(words):
    unigram_counts = Counter(words)
    bigram_counts = Counter(zip(words, words[1:]))
    return unigram_counts, bigram_counts

def total_bigrams(bigram_counts):
    return sum(bigram_counts.values())

def calculate_pmi(unigram_counts, bigram_counts, total_unigrams, total_bigrams):
    pmi_scores = defaultdict(float)
    
    for (w1, w2), bigram_count in bigram_counts.items():
        p_w1 = unigram_counts[w1] / total_unigrams
        p_w2 = unigram_counts[w2] / total_unigrams
        p_w1_w2 = bigram_count / total_bigrams
        
        if p_w1_w2 > 0:
            pmi_scores[(w1, w2)] = math.log2(p_w1_w2 / (p_w1 * p_w2))
    
    return pmi_scores

def calculate_chi_square(unigram_counts, bigram_counts, total_unigrams, total_bigrams):
    chi_square_scores = defaultdict(float)

    for (w1, w2), bigram_count in bigram_counts.items():
        expected_bigram_count = (unigram_counts[w1] * unigram_counts[w2]) / total_unigrams
        if expected_bigram_count > 0:
            chi_square_scores[(w1, w2)] = (bigram_count - expected_bigram_count) ** 2 / expected_bigram_count
    
    return chi_square_scores

def print_top_bigrams(score_dict):

    top_bigrams = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    
    for bigram, score in top_bigrams:
        print(f"{bigram[0]} {bigram[1]} {score:.4f}")

def main():

    corpus_file = "Collocations"
    measure_type = sys.argv[2].lower()

    words = read_corpus(corpus_file)

    unigram_counts, bigram_counts = calculate_counts(words)
    total_unigrams = sum(unigram_counts.values())
    total_bigrams_count = total_bigrams(bigram_counts)

    if measure_type == "pmi":
        pmi_scores = calculate_pmi(unigram_counts, bigram_counts, total_unigrams, total_bigrams_count)
        print_top_bigrams(pmi_scores)
    elif measure_type == "chi-square":
        chi_square_scores = calculate_chi_square(unigram_counts, bigram_counts, total_unigrams, total_bigrams_count)
        print_top_bigrams(chi_square_scores)
    else:
        print("Invalid measure. Choose either 'chi-square' or 'PMI'.")

if __name__ == "__main__":
    main()
