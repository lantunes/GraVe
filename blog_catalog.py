import argparse
import networkx as nx

NODES_FILE = "resources/BlogCatalog/data/nodes.csv"
EDGES_FILE = "resources/BlogCatalog/data/edges.csv"
GROUPS_FILE = "resources/BlogCatalog/data/group-edges.csv"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the BlogCatalog .adjlist and .gpickle files.')
    parser.add_argument('--name', '-n', action='store',
                        default=None,
                        help='The name of the files to be written (without the extension).')
    args = parser.parse_args()

    G = nx.Graph()

    with open(NODES_FILE) as nodes_file, open(EDGES_FILE) as edges_file, open(GROUPS_FILE) as groups_file:
        for node_line in nodes_file.readlines():
            node = node_line.strip()
            G.add_node(node)

        for edge_line in edges_file.readlines():
            edge = edge_line.strip()
            node1, node2 = edge.split(",")
            G.add_edge(node1, node2)

        for groups_line in groups_file.readlines():
            group_for_node = groups_line.strip()
            node_with_group, group = group_for_node.split(",")
            G.nodes[node_with_group]["group"] = group

    nx.write_adjlist(G, args.name + ".adjlist")
    nx.write_gpickle(G, args.name + ".gpickle")
