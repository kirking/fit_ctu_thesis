from itertools import count
import networkx as nx
import pandas as pd
import numpy as np
import random
from read_graph import read_graphs_in_networkx,save_graphs_nx

from stellargraph.core.graph import StellarDiGraph
import stellargraph as sg
from stellargraph import globalvar
from stellargraph import datasets
import matplotlib.pyplot as plt

encoding_dicts = {'citeseer': {'IR': 0, 'ML': 1, 'HCI': 2, 'DB': 3, 'AI': 4, 'Agents': 5},
 'cora' : {'Case_Based': 0, 'Neural_Networks': 1, 'Theory': 2, 'Rule_Learning': 3, 'Probabilistic_Methods': 4, 'Reinforcement_Learning': 5, 'Genetic_Algorithms': 6},
 'ENZYMES': {'1': 1,'2': 2,'3': 3}, 'MOL': {'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}}


# %matplotlib inline
def encode_labels(graph, encoding_dict, label_name = 'label', edge_label = None, cut = False):
  g = graph.copy()
  for idx, dic in g.node(data=True):
    tmp_val = dic[label_name]

    if cut:
      tmp_val = tmp_val[0]

    dic['topic'] = int(encoding_dict[tmp_val])
    del g.node[idx][label_name]

  if edge_label is not None:
    for u,v, dic in g.edges(data=True):
      tmp_val = dic[edge_label]
      dic['weight'] = tmp_val
      del g.edges[u,v][edge_label]

  return g

def draw_graph(g, filename, node_labelname = 'node_label', edge_labelname = 'weight'):
    print(g.edges(data=True))
    print(g.node(data=True))

    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    groups = set(nx.get_node_attributes(g,node_labelname).values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = g.nodes()

    colors = [g.node[n][node_labelname] for n in nodes]
# drawing nodes and edges separately so we can capture collection for colobar
    pos = nx.spring_layout(g,scale=0.01)

    ec = nx.draw_networkx_edges(g, pos, alpha=0.2)

    nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors, 
                            with_labels=False, node_size=20, cmap=plt.cm.jet)
    
    plt.colorbar(nc)

    plt.savefig(filename)
    plt.show()
    
class graph_loader:

  def load(filename, subgraph_count=100000, random_sampling = False, train_test_split = False):
    print(filename)
    (graphs,node_dict,edge_dict,node_label_freq_dict,edge_label_freq_dict) = read_graphs_in_networkx(filename,True,subgraph_count, random_sampling)
    print("Number of graphs loaded " , len(graphs))
    if train_test_split:
      train_graph = graphs[0]
      test_ind = 0
      print("TRAIN", int(len(graphs)*0.75))
      for indx in range(1, int(len(graphs)*0.75)):
        train_graph = nx.disjoint_union(train_graph, graphs[indx])
        test_ind = indx
      test_graph = graphs[test_ind + 1]
      for indx in range(test_ind + 1, len(graphs)):
        test_graph = nx.disjoint_union(test_graph, graphs[indx])
      
      return train_graph, test_graph
    else:
      j_graph = graphs[0]
      for indx in range(1, len(graphs)):
        j_graph = nx.disjoint_union(j_graph, graphs[indx])
        
    return j_graph

  def load_pickle(path_to_graphs, graphs_in_dir, batch_size, shuffle = False, encoding_dict = 'citeseer', node_label = 'label', edge_label = None, cut_label = False):
    
    graphs_indx = list(range(0, graphs_in_dir))
    
    if shuffle:
      random.shuffle(graphs_indx)

    G = nx.Graph()

    for i in range(batch_size):
      G = nx.disjoint_union(G, nx.read_gpickle(str(path_to_graphs) + str(graphs_indx[i]) + ".dat"))

    return encode_labels(G, encoding_dicts[encoding_dict], node_label, edge_label, cut_label)

  def transform_to_sg(graph_n, label_name = 'node_label', edge_label = None):
    
    graph = graph_n.copy()

    labels = [n[1][label_name] for n in graph.nodes.data()]

    indexes = [n[0] for n in graph.nodes.data()]

    features_df = pd.DataFrame({"topic": labels}, index = indexes )
    if edge_label is not None:
      graph = sg.StellarGraph.from_networkx(graph, node_features = features_df, edge_features = 'weight')
    graph = sg.StellarGraph.from_networkx(graph, node_features = features_df)

    return graph, features_df


  def draw(graph, filename):
    draw_graph(graph, filename)

  
