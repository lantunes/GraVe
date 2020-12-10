from grave import FactorizationMachine
import networkx as nx


if __name__ == '__main__':

    G = nx.read_gpickle("examples/cora.gpickle")
    feature_dict = {}
    for node in G:
        feature_dict[node] = G.nodes[node]["attr"]

    fm = FactorizationMachine(dim=128, y_max=100, alpha=0.75, context_window_size=10)

    X, Y = fm.build_training_data("examples/cora.node2vec.walks", feature_dict, workers=2)

    fm.save("cora.fm.training.model")
    FactorizationMachine.save_training_data(X, Y, fm.dictionary, "cora.fm.training.data", sparsify=True)
