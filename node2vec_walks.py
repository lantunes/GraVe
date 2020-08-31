import argparse
import networkx as nx

from grave import Node2VecGraph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph filename')
    parser.add_argument('--output', nargs='?', required=True,
                        help='Output walks filename (no extension)')
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    args = parser.parse_args()
    G = nx.read_adjlist(args.input)
    # assumes an unweighted graph
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    # assumes an undirected graph
    G = G.to_undirected()

    n2v_G = Node2VecGraph(G, False, args.p, args.q)

    n2v_G.preprocess_transition_probs()
    walks = n2v_G.simulate_walks(args.num_walks, args.walk_length)

    with open(args.output + ".node2vec.walks" , "w") as output:
        for walk in walks:
            line = " ".join(walk)
            output.write(line + "\n")
        output.flush()
