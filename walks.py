import random
from argparse import ArgumentParser

from deepwalk import graph
from deepwalk import walks as serialized_walks

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')
    parser.add_argument('--output', required=True,
                        help='file containing the walks (i.e. corpus)')
    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--walk-length', default=40, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random walk generator.')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')
    args = parser.parse_args()
    G = graph.load_adjacencylist(args.input, undirected=args.undirected)

    print("Number of nodes: {}".format(len(G.nodes())))
    num_walks = len(G.nodes()) * args.number_walks
    print("Number of walks: {}".format(num_walks))
    data_size = num_walks * args.walk_length
    print("Data size (walks*length): {}".format(data_size))

    walks_filebase = args.output + ".walks"
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                                      path_length=args.walk_length, alpha=0,
                                                      rand=random.Random(args.seed),
                                                      num_workers=args.workers)