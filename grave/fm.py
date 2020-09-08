import numpy as np

from autograd import grad
from autograd.misc.optimizers import adam

from multiprocessing import Pool

from networkx.utils import open_file

from scipy import sparse

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

        self._curr_loss = 0

    def build_training_data(self, corpus, features_dict, workers=1):
        """
        Constructs the training data. NOTE: only binary feature vectors are currently supported.
        :param corpus: a list of lists, where each item in an inner list is a word, or the path to the file containing
                      the corpus, where each line is a sentence, and each word on each line is space separated
        :param features_dict: a map of all words to their feature vectors
        :param workers: the number of processors to use in parallel (must be >= 1)
        :return: the feature matrix, X, and the target labels (i.e. occurrence counts), Y
        """

        # construct self.dictionary using features_dict (embeddings list index values start from 0)
        self.dictionary = self._make_dictionary(features_dict)

        if workers > 1:
            return self._build_training_data_parallel(corpus, features_dict, workers)

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

        return self._counts_to_data(feature_vector_counts, self._fv_length(features_dict))

    def _fv_length(self, features_dict):
        return len(features_dict)+len(next(iter(features_dict.values())))

    def _build_training_data_parallel(self, corpus, features_dict, workers):
        """
        The corpus is read into memory and partitioned evenly amongst the workers, who then compute their
        own feature vector count maps. All the worker feature vector count maps are merged at the end.
        """
        #
        feature_vector_counts = {}
        data = []
        if type(corpus) is str:
            with open(corpus, "r") as corpus_file:
                for line in corpus_file.readlines():
                    data.append(line.strip().split(" "))
        else:
            data = corpus
        chunks = np.array_split(np.array(data), workers)

        with Pool(processes=workers) as pool:
            args = []
            for chunk in chunks:
                args.append((features_dict, chunk))
            for r in pool.starmap(self._run, args):
                # merge the feature vector counts from all workers
                for fv in list(r.keys()):
                    if fv not in feature_vector_counts:
                        feature_vector_counts[fv] = r[fv]
                    else:
                        feature_vector_counts[fv] += r[fv]
                    del r[fv]

        return self._counts_to_data(feature_vector_counts, self._fv_length(features_dict))

    def _run(self, features_dict, chunk):
        fv_counts = {}
        for words in chunk:
            self._count_feature_vectors(words, features_dict, fv_counts)
        return fv_counts

    def _counts_to_data(self, feature_vector_counts, fv_size):
        X = []
        Y = []
        for fv in feature_vector_counts:
            X.append(self._string_to_fv(fv, fv_size))
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
            loss = self._loss(params, X[idx], Y[idx])
            self._curr_loss += loss
            return loss

        # Get gradient of objective using autograd.
        objective_grad = grad(objective)

        def print_progress(params, iter, gradient):
            if (iter+1) % num_batches == 0:
                print("epoch: {:7}, loss: {:15}".format((iter // num_batches) + 1, self._curr_loss._value))
                self._curr_loss = 0

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
            nz = x.nonzero()[0]
            for nz_i in range(len(nz)):
                for nz_j in range(nz_i + 1, len(nz)):
                    i = nz[nz_i]
                    j = nz[nz_j]
                    interaction_score += x[i] * x[j] * np.dot(W[i], W[j])
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
        return self._fv_to_string(np.concatenate([fv, combined]))

    def _fv_to_string(self, fv):
        return ",".join(map(str, fv.nonzero()[0]))

    def _string_to_fv(self, string, size):
        fv = np.zeros(size)
        for i in [int(s) for s in string.split(",")]:
            fv[i] = 1
        return fv

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
    def save_training_data(X, Y, dictionary, path, protocol=pickle.HIGHEST_PROTOCOL, sparsify=False):
        """
        Serializes the given data with pickle to the given path.
        :param X: the feature vectors to be saved
        :param Y: the feature vector corresponding count labels to be saved
        :param dictionary: a map of node id to feature vector index
        :param path: the filename of the generated pickle file
        :param protocol: the pickle protocol
        :param sparsify: whether to convert X to a sparse matrix before serializing
        """
        if sparsify:
            X = sparse.csr_matrix(X)
        data = (X, Y, dictionary)
        pickle.dump(data, path, protocol)

    @staticmethod
    def save_training_data_csv(X, Y, dictionary, data_path, dict_path):
        """
        Saves the data into .csv files. The first column in the data CSV file will be
        the label (Y), and the remaining columns will be the features (X).
        :param X: the feature vectors to be saved
        :param Y: the feature vector corresponding count labels to be saved
        :param dictionary: a map of node id to feature vector index
        :param data_path: the filename of the training data to be saved
        :param dict_path: the filename of the dictionary to be saved
        """
        with open(data_path, "w") as data_file:
            for i, x in enumerate(X):
                x.insert(0, Y[i])
                data_file.write(",".join(map(str, x)) + "\n")
            data_file.flush()
        with open(dict_path, "w") as dict_file:
            for item in dictionary.items():
                dict_file.write(",".join(map(str, item)) + "\n")
            dict_file.flush()

    @staticmethod
    @open_file(0, mode='rb')
    def load_training_data(path, sparse=False):
        X, Y, dictionary = pickle.load(path)
        if sparse:
            X = X.toarray()
        return X, Y, dictionary

    @staticmethod
    def load_training_data_csv(data_path, dict_path):
        X = []
        Y = []
        with open(data_path, "r") as data_file:
            for line in data_file.readlines():
                data = line.strip().split(",")
                Y.append(int(data[0]))
                X.append([float(x) for x in data[1:]])
        dictionary = {}
        with open(dict_path, "r") as dict_file:
            for line in dict_file.readlines():
                k, v = line.strip().split(",")
                dictionary[k] = int(v)
        return X, Y, dictionary