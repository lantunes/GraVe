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


def where_greater_feature_combiner(fv_word1, fv_word2):
    """
    Returns a vector containing the greatest of the two values between fv_word1 and fv_word2 at each position.
    :param fv_word1: the feature vector for the first word of the co-occurrence pair
    :param fv_word2: the feature vector for the second word of the co-occurrence pair
    :return: a single feature vector representing word1 and word2
    """
    a = np.array(fv_word1, copy=False)
    b = np.array(fv_word2, copy=False)
    return np.where(a > b, a, b)
