import argparse, os
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
import networkx as nx
from data.generator.HouseConfig import getRoomName
import matplotlib.pyplot as plt


def main(args):
    collection_directory = args.collection_directory
    count = args.count

    graph_directory = f'{collection_directory}/room-classification/graph'
    floor_directory = f'{collection_directory}/symbol-detection/resized'

    data = []
    for file in os.scandir(graph_directory):
        if file.is_file():
            data.append(file.name)

    random_data = np.random.choice(data, count)

    for datum in random_data:
        ID = datum.split('.')[0]

        graph_location = f'{graph_directory}/{ID}.pt'
        floor_location = f'{floor_directory}/{ID}.png'

        graph = torch.load(graph_location)
        if not isinstance(graph, Data):
            continue

        y = np.argmax(graph.y.numpy(), axis=1)
        edge_attr = np.argmax(graph.edge_attr.numpy(), axis=1)
        edge_index = graph.edge_index.numpy()

        G = nx.DiGraph()

        nodes = list(range(len(y)))
        G.add_nodes_from(nodes)
        G.add_edges_from(edge_index.T)

        labels = {}
        for node in nodes:
            labels[node] = getRoomName(int(y[node]))
        
        # nx.draw_networkx(G, labels=labels, arrows=True)
        # plt.title(f"{ID} room graph")
        # plt.savefig(f"{ID}.jpeg", dpi = 300)

        print(graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Graph Visualizer',
        description = 'Visualize graph data, correspond to floorplan data. Only for internal usage.',
        epilog = 'Created by Christian Budhi Sabdana aka MaclaurinSeries (GitHub)'
    )
    parser.add_argument('-d', '--collection-dir',
                        type=str,
                        default='.\\input',
                        help='dataset input directory',
                        dest='collection_directory')
    parser.add_argument('-c', '--count',
                        type=int,
                        default=5,
                        help='number of graph to be displayed',
                        dest='count')
    
    main(parser.parse_args())