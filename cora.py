import argparse
import networkx as nx

CORA_CITES = "resources/Cora/data/cora.cites"
CORA_CONTENT = "resources/Cora/data/cora.content"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the Cora .adjlist and .gpickle files.')
    parser.add_argument('--name', '-n', action='store',
                        default=None,
                        help='The name of the files to be written (without the extension).')
    args = parser.parse_args()

    G = nx.DiGraph()

    with open(CORA_CITES) as cites_file, open(CORA_CONTENT) as content_file:
        for cites_line in cites_file:
            cites_line = cites_line.strip()
            node_to, node_from = cites_line.split("\t")
            G.add_edge(node_from, node_to)

        for content_line in content_file:
            content_line = content_line.strip()
            content = content_line.split("\t")
            node = content[0]
            label = content[-1]
            attribute = [int(x) for x in content[1:-1]]
            G.nodes[node]["label"] = label
            G.nodes[node]["attr"] = attribute

    nx.write_adjlist(G, args.name + ".adjlist")
    nx.write_gpickle(G, args.name + ".gpickle")
