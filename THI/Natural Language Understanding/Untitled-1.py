class NGramLM:
    def __init__(self, corpus, max_order):
        self.max_order = max_order
        self.ngram_models = [{} for _ in range(max_order)]
        self.total_counts = [0 for _ in range(max_order)]
        self.train(corpus)

    def train(self, corpus):
        for n in range(1, self.max_order + 1):
            for i in range(len(corpus) - n + 1):
                ngram = tuple(corpus[i:i+n])
                self.ngram_models[n-1][ngram] = self.ngram_models[n-1].get(ngram, 0) + 1
                self.total_counts[n-1] += 1

    def probability(self, ngram):
        for n in range(self.max_order, 0, -1):
            if ngram[:n] in self.ngram_models[n-1]:
                return self.ngram_models[n-1][ngram[:n]] / self.total_counts[n-1]
        return 0  # If the ngram is not found in any model, return 0

# Example usage:
corpus = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
lm = NGramLM(corpus, 3)
print(lm.probability(("the", "quick")))  # Output: Probability of the bigram ("the", "quick")
