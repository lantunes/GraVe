from grave import FactorizationMachine
import networkx as nx

import matplotlib.pyplot as plt


def get_community_color(node):
    if node in ["11", "5", "6", "7", "17"]:
        return "lightblue"
    if node in ["12", "22", "18", "1", "20", "2", "14", "8", "4", "13"]:
        return "red"
    if node in ["3", "10", "32", "29", "28", "26", "25"]:
        return "lightgreen"
    else:
        return "purple"


def get_label_color(node, G):
    club = G.nodes[node]['club']
    if club == "Mr. Hi":
        return "k"
    else:
        return "y"


if __name__ == '__main__':

    G = nx.karate_club_graph()
    feature_dict = {}
    for node in G:
        club = G.nodes[node]['club']
        feature_dict[str(node+1)] = [1, 0] if club == "Mr. Hi" else [0, 1]

    fm = FactorizationMachine(dim=2, y_max=100, alpha=0.75, context_window_size=1)
    X, Y = fm.build_training_data("examples/karate2.walks.0", feature_dict)
    fm.fit(X, Y, batch_size=1, learning_rate=0.01, num_epochs=10)

    color_map = []
    X_ = []
    Y_ = []

    for node in G:
        node = str(node+1)
        color_map.append(get_community_color(node))
        embedding = fm.W[fm.dictionary[node]]
        X_.append(embedding[0])
        Y_.append(embedding[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_, Y_, color=color_map)
    for vertex in fm.dictionary.keys():
        embedding = fm.W[fm.dictionary[vertex]]
        c = get_label_color(int(vertex)-1, G)
        ax.annotate(vertex, (embedding[0], embedding[1]), color=c)
    plt.figure()
    nx.draw(G, with_labels=True, node_color=color_map, labels={v: str(v+1) for v in G.nodes})
    plt.show()

