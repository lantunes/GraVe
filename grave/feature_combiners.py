import numpy as np


def addition_feature_combiner(word1, word2, feature_dict):
    """
    Returns the element-wise addition of the feature vectors for each word.
    :param word1: the first word of the co-occurrence pair
    :param word2: the second word of the co-occurrence pair
    :param feature_dict: a map of words to their feature vectors
    :return: a single feature vector representing word1 and word2
    """
    if feature_dict[word1] is None or len(feature_dict[word1]) == 0 or \
            feature_dict[word2] is None or len(feature_dict[word2]) == 0:
        return []
    return np.add(feature_dict[word1], feature_dict[word2])


def bitwise_or_feature_combiner(word1, word2, feature_dict):
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
