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


if __name__ == '__main__':

    G = nx.readwrite.read_adjlist("examples/karate2.adjlist")

    f = FactorizationMachine(dim=2, y_max=100, alpha=0.75, context_window_size=1)
    feature_dict = {
        "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": [], "11": [], "12": [],
        "13": [], "14": [], "15": [], "16": [], "17": [], "18": [], "19": [], "20": [], "21": [], "22": [], "23": [],
        "24": [], "25": [], "26": [], "27": [], "28": [], "29": [], "30": [], "31": [], "32": [], "33": [], "34": []
    }
    X, Y = f.build_training_data("examples/karate2.walks.0", feature_dict)
    f.fit(X, Y, batch_size=1, learning_rate=0.01, num_epochs=10, use_autograd=False)

    color_map = []
    X_ = []
    Y_ = []

    for node in G:
        color_map.append(get_community_color(node))
        embedding = f.W[f.dictionary[node]]
        X_.append(embedding[0])
        Y_.append(embedding[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_, Y_, color=color_map)
    for vertex in f.dictionary.keys():
        embedding = f.W[f.dictionary[vertex]]
        ax.annotate(vertex, (embedding[0], embedding[1]))
    plt.figure()
    nx.draw(G, with_labels=True, node_color=color_map)
    plt.show()