import numpy as np

from autograd import grad
from autograd.misc.optimizers import adam

from networkx.utils import open_file

try:
    import cPickle as pickle
except ImportError:
    import pickle


class FactorizationMachine:
    """
    A degree-2 Factorization Machine.
    """
    def __init__(self, dim, y_max, alpha, context_window_size, dictionary=None, feature_combiner=None, W=None, b=None):
        # a map of words to their embeddings list index
        self.dictionary = {} if dictionary is None else dictionary
        # the embedding matrix
        self.W = W
        # the embedding biases used during training
        self.b = b
        # the dimensionality of the learned embeddings
        self.dim = dim
        # the y_max hyperparameter
        self.y_max = y_max
        # the alpha hyperparameter
        self.alpha = alpha
        # the symmetric context window size for which to determine co-occurrences
        self.context_window_size = context_window_size
        # a function that determines how a single feature vector can be constructed for two words
        self.feature_combiner = self._default_feature_combiner if feature_combiner is None else feature_combiner

    def build_training_data(self, corpus, features_dict):
        """
        Constructs the training data.
        :param corpus: a list of lists, where each item in an inner list is a word, or the path to the file containing
                      the corpus, where each line is a sentence, and each word on each line is space separated
        :param features_dict: a map of all words to their feature vectors
        :return: the feature matrix, X, and the target labels (i.e. occurrence counts), Y
        """

        # construct self.dictionary using features_dict (embeddings list index values start from 0)
        self.dictionary = self._make_dictionary(features_dict)

        # construct a map of tuple feature vectors to counts (i.e. (1,0,1,0,0) -> 3)
        #  as we are sliding the context window over each line
        feature_vector_counts = {}
        if type(corpus) is str:
            with open(corpus, "r") as corpus_file:
                for line in corpus_file.readlines():
                    words = line.strip().split(" ")
                    self._count_feature_vectors(words, features_dict, feature_vector_counts)
        else:
            for words in corpus:
                self._count_feature_vectors(words, features_dict, feature_vector_counts)

        X = []
        Y = []
        for fv in feature_vector_counts:
            X.append(list(fv))
            Y.append(feature_vector_counts[fv])

        return X, Y

    def _init_params(self, num_embeddings):
        word_vectors = (np.random.rand(num_embeddings, self.dim) - 0.5) / self.dim
        word_biases = np.zeros(num_embeddings, dtype=np.float64)
        return word_vectors, word_biases

    def fit(self, X, Y, batch_size=100, num_epochs=3, learning_rate=0.001):
        """
        Minimize the loss.
        """
        num_embeddings = len(X[0])
        print("num embeddings: %s" % num_embeddings)
        print("num feature vectors: %s" % len(X))
        print("num count labels: %s" % len(Y))

        X = np.array(X)
        Y = np.array(Y)

        init_params = self._init_params(num_embeddings)
        num_batches = int(np.ceil(len(X) / batch_size))

        print("num batches: %s" % num_batches)

        def batch_indices(iter):
            idx = iter % num_batches
            return slice(idx * batch_size, (idx + 1) * batch_size)

        def objective(params, iter):
            idx = batch_indices(iter)
            return self._loss(params, X[idx], Y[idx])

        # Get gradient of objective using autograd.
        objective_grad = grad(objective)

        def print_progress(params, iter, gradient):
            if iter % num_batches == 0:
                print("epoch: {:7}".format((iter // num_batches) + 1))

        optimized_params = adam(objective_grad, init_params, step_size=learning_rate,
                                num_iters=num_epochs * num_batches, callback=print_progress)

        self.W, self.b = optimized_params

    def _loss(self, params, X, Y):
        W = params[0]
        b = params[1]
        return np.sum(self._weight(Y) * (self._score(X, W, b) - np.log(Y))**2)

    def _weight(self, Y):
        return np.where(Y < self.y_max, (Y / self.y_max)**self.alpha, 1.0)

    def _score(self, X, W, b):
        """
        :param x: feature vectors
        :return: an array of scalar scores for each feature vector
        """
        scores = []
        for k in range(len(X)):
            x = X[k]
            bias_score = np.dot(x, b)
            interaction_score = 0
            for i in range(len(x)):
                for j in range(len(x)):
                    if i == j:
                        continue
                    prod = x[i] * x[j]
                    if prod != 0:
                        interaction_score += prod * np.dot(W[i], W[j])
            scores.append(bias_score + interaction_score)
        return np.array(scores)

    def _default_feature_combiner(self, word1, word2, feature_dict):
        """
        Returns the bitwise OR of the feature vectors for each word.
        :param word1: the first word of the co-occurrence pair
        :param word2: the second word of the co-occurrence pair
        :param feature_dict: a map of words to their feature vectors
        :return: a single feature vector representing word1 and word2
        """
        if feature_dict[word1] is None or len(feature_dict[word1]) == 0 or \
                feature_dict[word2] is None or len(feature_dict[word2]) == 0:
            return []
        return np.bitwise_or(feature_dict[word1], feature_dict[word2])

    def _make_dictionary(self, features_dict):
        dict = {}
        i = 0
        for word in features_dict:
            dict[word] = i
            i += 1
        return  dict

    def _count_feature_vectors(self, words, features_dict, feature_vector_counts):
        slices = self._get_slices(words, self.context_window_size)
        for target, context_words in slices:
            for context_word in context_words:
                if target == context_word:
                    continue
                # construct the feature vector for this pair
                feature_vector = self._init_feature_vector(target, context_word, features_dict)
                if feature_vector not in feature_vector_counts:
                    feature_vector_counts[feature_vector] = 0
                feature_vector_counts[feature_vector] += 1

    def _get_slices(self, seq, window):
        slices = []
        for i in range(len(seq)):
            target = seq[i]
            context_words = []
            for j in range(1, window+1):
                # left of target
                left_idx = i - j
                if left_idx > -1:
                    context_words.append(seq[left_idx])
                # right of target
                right_idx = i + j
                if right_idx <= len(seq) - 1:
                    context_words.append(seq[right_idx])
            slices.append((target, context_words))
        return slices

    def _init_feature_vector(self, word1, word2, feature_dict):
        fv = np.zeros(len(feature_dict))
        fv[self.dictionary[word1]] = 1
        fv[self.dictionary[word2]] = 1
        combined = self.feature_combiner(word1, word2, feature_dict)
        return tuple(np.concatenate([fv, combined]))

    @open_file(1, mode='wb')
    def save(self, path, protocol=pickle.HIGHEST_PROTOCOL):
        data = (self.W, self.b, self.dictionary, self.dim, self.y_max, self.alpha,
                self.context_window_size, self.feature_combiner)
        pickle.dump(data, path, protocol)

    @staticmethod
    @open_file(0, mode='rb')
    def load_model(path):
        W, b, dictionary, dim, y_max, alpha, context_window_size, feature_combiner = pickle.load(path)
        return FactorizationMachine(dim=dim, y_max=y_max, alpha=alpha, context_window_size=context_window_size,
                                    dictionary=dictionary,feature_combiner=feature_combiner, W=W, b=b)

    @staticmethod
    @open_file(3, mode='wb')
    def save_training_data(X, Y, dictionary, path, protocol=pickle.HIGHEST_PROTOCOL):
        data = (X, Y, dictionary)
        pickle.dump(data, path, protocol)

    @staticmethod
    @open_file(0, mode='rb')
    def load_training_data(path):
        return pickle.load(path)
