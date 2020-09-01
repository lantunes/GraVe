import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from glove import Glove
import networkx as nx


def color_by_label(label):
    if label == "Case_Based":
        return "red"
    if label == "Genetic_Algorithms":
        return "blue"
    if label == "Neural_Networks":
        return "green"
    if label == "Probabilistic_Methods":
        return "yellow"
    if label == "Reinforcement_Learning":
        return "purple"
    if label == "Rule_Learning":
        return "brown"
    if label == "Theory":
        return "gray"
    raise Exception("Unknown label: %s" % label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a t-SNE plot for a trained GloVe model.')
    parser.add_argument('--model', '-m', action='store',
                        required=True,
                        help='The filename of the stored GloVe model.')
    parser.add_argument('--graph', '-g', action='store',
                        required=True,
                        help='The filename of the stored .gpickle file for the graph.')
    parser.add_argument('--perplexity', '-p', action='store',
                        default=50, type=int,
                        help='The perplexity parameter for t-SNE.')
    parser.add_argument('--iterations', '-i', action='store',
                        default=500, type=int,
                        help='The number of iterations for t-SNE.')
    parser.add_argument('--learning-rate', '-r', action='store',
                        default=10, type=int,
                        help='The learning rate parameter for t-SNE.')

    args = parser.parse_args()

    G = nx.read_gpickle(args.graph)
    glove = Glove.load(args.model)

    color_map = []
    X = []
    for node in G:
        color_map.append(color_by_label(G.nodes[node]['label']))
        X.append(glove.word_vectors[glove.dictionary[node]])

    tsne = TSNE(n_components=2, verbose=1,
                perplexity=args.perplexity, n_iter=args.iterations, learning_rate=args.learning_rate)
    result = tsne.fit_transform(X)

    plt.scatter(result[:, 0], result[:, 1], c=color_map, edgecolors='black')
    plt.show()