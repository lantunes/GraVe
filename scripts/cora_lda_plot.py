import matplotlib.pyplot as plt
import networkx as nx
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D

"""
Here, we use LDA for dimensionality reduction, reducing the size of the Cora feature vector from 1,433 to 3.
"""

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
    G = nx.read_gpickle("../examples/cora.gpickle")

    color_map = []
    X = []
    Y = []
    for node in G:
        color_map.append(color_by_label(G.nodes[node]['label']))
        X.append(G.nodes[node]['attr'])
        Y.append(G.nodes[node]['label'])

    result = LinearDiscriminantAnalysis(n_components=3).fit_transform(X, Y)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(result[:, 0], result[:, 1], result[:, 2], c=color_map)
    plt.show()
