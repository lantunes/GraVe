import numpy as np


def addition_feature_combiner(fv_word1, fv_word2):
    """
    Returns the element-wise addition of the feature vectors for each word.
    :param fv_word1: the feature vector for the first word of the co-occurrence pair
    :param fv_word2: the feature vector for the second word of the co-occurrence pair
    :return: a single feature vector representing word1 and word2
    """
    return np.add(fv_word1, fv_word2)


def bitwise_or_feature_combiner(fv_word1, fv_word2):
    """
    Returns the bitwise OR of the feature vectors for each word.
    NOTE: if this feature combiner is used, then feature vectors must be binary vectors.
    :param fv_word1: the feature vector for the first word of the co-occurrence pair
    :param fv_word2: the feature vector for the second word of the co-occurrence pair
    :return: a single feature vector representing word1 and word2
    """
    return np.bitwise_or(fv_word1, fv_word2)
