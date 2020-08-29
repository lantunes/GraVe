import argparse

from glove import Glove

import matplotlib.pyplot as plt

import networkx as nx

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inspect a trained GloVe model.')
    parser.add_argument('--model', '-m', action='store',
                        required=True,
                        help='The filename of the stored GloVe model.')
    parser.add_argument('--adjlist', '-a', action='store',
                        default="",
                        help='The filename containing the adjacency list')

    args = parser.parse_args()

    glove = Glove.load(args.model)

    # the embeddings for the entire vocabulary
    print(glove.word_vectors)

    # the embedding for vertex "1"
    print(glove.word_vectors[glove.dictionary['1']])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(glove.word_vectors[:,0], glove.word_vectors[:,1])
    for vertex in glove.dictionary.keys():
        embedding = glove.word_vectors[glove.dictionary[vertex]]
        ax.annotate(vertex, (embedding[0], embedding[1]))

    if args.adjlist:
        plt.figure()
        G = nx.readwrite.read_adjlist(args.adjlist)
        nx.draw(G, with_labels=True)
        plt.show()