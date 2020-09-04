import unittest

from grave import FactorizationMachine


class TestFactorizationMachine(unittest.TestCase):

    def test_build_training_data(self):
        corpus = [
            ["1", "2", "3", "4", "5"],
            ["3", "1", "4"],
            ["5", "2", "1", "5", "4", "2", "1", "3"],
            ["2"],
            [],
            ["1", "1"],
            ["1", "4"]
        ]

        features_dict = {
            "1": [1, 0, 0],
            "2": [0, 1, 0],
            "3": [0, 0, 1],
            "4": [1, 1, 0],
            "5": [1, 0, 1],
        }

        fm = FactorizationMachine(dim=10, y_max=1, alpha=1, context_window_size=2)

        X, Y = fm.build_training_data(corpus, features_dict)

        self.assertEqual(len(X), 10)
        self.assertEqual(len(Y), 10)

        fv_1_2, count_1_2 = self._get_fv("1", "2", X, Y, fm)
        fv_1_3, count_1_3 = self._get_fv("1", "3", X, Y, fm)
        fv_1_4, count_1_4 = self._get_fv("1", "4", X, Y, fm)
        fv_1_5, count_1_5 = self._get_fv("1", "5", X, Y, fm)
        fv_2_3, count_2_3 = self._get_fv("2", "3", X, Y, fm)
        fv_2_4, count_2_4 = self._get_fv("2", "4", X, Y, fm)
        fv_2_5, count_2_5 = self._get_fv("2", "5", X, Y, fm)
        fv_3_4, count_3_4 = self._get_fv("3", "4", X, Y, fm)
        fv_3_5, count_3_5 = self._get_fv("3", "5", X, Y, fm)
        fv_4_5, count_4_5 = self._get_fv("4", "5", X, Y, fm)

        self._assert_feature_vector("1", "2", fv_1_2, features_dict, fm)
        self._assert_feature_vector("1", "3", fv_1_3, features_dict, fm)
        self._assert_feature_vector("1", "4", fv_1_4, features_dict, fm)
        self._assert_feature_vector("1", "5", fv_1_5, features_dict, fm)
        self._assert_feature_vector("2", "3", fv_2_3, features_dict, fm)
        self._assert_feature_vector("2", "4", fv_2_4, features_dict, fm)
        self._assert_feature_vector("2", "5", fv_2_5, features_dict, fm)
        self._assert_feature_vector("3", "4", fv_3_4, features_dict, fm)
        self._assert_feature_vector("3", "5", fv_3_5, features_dict, fm)
        self._assert_feature_vector("4", "5", fv_4_5, features_dict, fm)

        self.assertEqual(count_1_2, 6)
        self.assertEqual(count_1_3, 6)
        self.assertEqual(count_1_4, 8)
        self.assertEqual(count_1_5, 4)
        self.assertEqual(count_2_3, 4)
        self.assertEqual(count_2_4, 4)
        self.assertEqual(count_2_5, 6)
        self.assertEqual(count_3_4, 4)
        self.assertEqual(count_3_5, 2)
        self.assertEqual(count_4_5, 4)

    def test_build_training_data_parallel(self):
        corpus = [
            ["1", "2", "3", "4", "5"],
            ["3", "1", "4"],
            ["5", "2", "1", "5", "4", "2", "1", "3"],
            ["2"],
            [],
            ["1", "1"],
            ["1", "4"]
        ]

        features_dict = {
            "1": [1, 0, 0],
            "2": [0, 1, 0],
            "3": [0, 0, 1],
            "4": [1, 1, 0],
            "5": [1, 0, 1],
        }

        fm = FactorizationMachine(dim=10, y_max=1, alpha=1, context_window_size=2)

        X, Y = fm.build_training_data(corpus, features_dict, workers=4)

        self.assertEqual(len(X), 10)
        self.assertEqual(len(Y), 10)

        fv_1_2, count_1_2 = self._get_fv("1", "2", X, Y, fm)
        fv_1_3, count_1_3 = self._get_fv("1", "3", X, Y, fm)
        fv_1_4, count_1_4 = self._get_fv("1", "4", X, Y, fm)
        fv_1_5, count_1_5 = self._get_fv("1", "5", X, Y, fm)
        fv_2_3, count_2_3 = self._get_fv("2", "3", X, Y, fm)
        fv_2_4, count_2_4 = self._get_fv("2", "4", X, Y, fm)
        fv_2_5, count_2_5 = self._get_fv("2", "5", X, Y, fm)
        fv_3_4, count_3_4 = self._get_fv("3", "4", X, Y, fm)
        fv_3_5, count_3_5 = self._get_fv("3", "5", X, Y, fm)
        fv_4_5, count_4_5 = self._get_fv("4", "5", X, Y, fm)

        self._assert_feature_vector("1", "2", fv_1_2, features_dict, fm)
        self._assert_feature_vector("1", "3", fv_1_3, features_dict, fm)
        self._assert_feature_vector("1", "4", fv_1_4, features_dict, fm)
        self._assert_feature_vector("1", "5", fv_1_5, features_dict, fm)
        self._assert_feature_vector("2", "3", fv_2_3, features_dict, fm)
        self._assert_feature_vector("2", "4", fv_2_4, features_dict, fm)
        self._assert_feature_vector("2", "5", fv_2_5, features_dict, fm)
        self._assert_feature_vector("3", "4", fv_3_4, features_dict, fm)
        self._assert_feature_vector("3", "5", fv_3_5, features_dict, fm)
        self._assert_feature_vector("4", "5", fv_4_5, features_dict, fm)

        self.assertEqual(count_1_2, 6)
        self.assertEqual(count_1_3, 6)
        self.assertEqual(count_1_4, 8)
        self.assertEqual(count_1_5, 4)
        self.assertEqual(count_2_3, 4)
        self.assertEqual(count_2_4, 4)
        self.assertEqual(count_2_5, 6)
        self.assertEqual(count_3_4, 4)
        self.assertEqual(count_3_5, 2)
        self.assertEqual(count_4_5, 4)

    def test_build_training_data_parallel2(self):
        corpus = [
            ["1", "2"],
            ["1", "2"],
            ["1", "2"],
            ["1", "2"],
            ["1", "2"],
            ["1", "2"],
            ["1", "2"],
            ["1", "2"],
        ]

        features_dict = {
            "1": [1, 0],
            "2": [0, 1],
        }

        fm = FactorizationMachine(dim=10, y_max=1, alpha=1, context_window_size=2)

        X, Y = fm.build_training_data(corpus, features_dict, workers=4)

        self.assertEqual(len(X), 1)
        self.assertEqual(len(Y), 1)

        fv_1_2, count_1_2 = self._get_fv("1", "2", X, Y, fm)

        self._assert_feature_vector("1", "2", fv_1_2, features_dict, fm)

        self.assertEqual(count_1_2, 16)

    def _get_fv(self, word1, word2, X, Y, fm):
        word1_id = fm.dictionary[word1]
        word2_id = fm.dictionary[word2]
        for i in range(len(X)):
            x = X[i]
            if x[word1_id] == 1.0 and x[word2_id] == 1.0:
                return x, Y[i]

    def _assert_feature_vector(self, word1, word2, feature_vector, features_dict, fm):
        word1_id = fm.dictionary[word1]
        word2_id = fm.dictionary[word2]
        num_words = len(features_dict)
        # check that the word bits are set properly
        for i in range(num_words):
            if i == word1_id or i == word2_id:
                self.assertEqual(feature_vector[i], 1.0)
            else:
                self.assertEqual(feature_vector[i], 0.0)
        # check that the feature bits are set properly
        word1_features = features_dict[word1]
        word2_features = features_dict[word2]
        for i in range(num_words, len(feature_vector)):
            word_features_idx = len(word1_features) - (len(feature_vector) - i)
            if word1_features[word_features_idx] == 1.0 or word2_features[word_features_idx] == 1.0:
                self.assertEqual(feature_vector[i], 1.0)
            else:
                self.assertEqual(feature_vector[i], 0.0)