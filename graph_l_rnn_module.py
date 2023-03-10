# -*- coding: utf-8 -*-
"""graph-l-rnn_nodule.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zYBZ13N5TM4xxcEWSyT0oXtVvUhOch40
"""

# Commented out IPython magic to ensure Python compatibility.
# from google.colab import drive
# drive.mount('/content/drive')

# %cd /content/drive/MyDrive/Github/GraphRNN/graph-label-rnn

# Commented out IPython magic to ensure Python compatibility.
# install StellarGraph if running on Google Colab
import sys


import networkx as nx  
import matplotlib.pyplot as plt
import operator
import numpy as np
import numpy
import sys
import json
import pdb
from sklearn.model_selection import train_test_split
import random
import sys
import pickle
import numpy as np
import random
np.set_printoptions(threshold=sys.maxsize)
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
from read_graph import read_graphs_in_networkx,save_graphs_nx
from utils import calculate_M,graphs_db,encode_M_matrix,decode_M_matrix
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from label_rnn_test import CUSTOM_RNN_NODE, CUSTOM_RNN_EDGE, pick_random_label, sample_multi, cut_graph

def graph_label_rnn_generate(out_file, graphs_count= 100, name = "epoch"):

  folder_param = "./models/"
  #epoch = 'epoch'
  epoch = name
  number_of_graphs  = graphs_count
  ofile = out_file
  model_parameters = pickle.load(open(folder_param+"parameters_"+str(epoch)+".pkl","rb"))


  M = model_parameters['M']
  hidden_size_node_rnn = model_parameters['hidden_size_node_rnn']
  hidden_size_edge_rnn = model_parameters['hidden_size_edge_rnn']
  embedding_size_node_rnn = model_parameters['embedding_size_node_rnn']
  embedding_size_edge_rnn = model_parameters['embedding_size_edge_rnn']
  num_layers = model_parameters['num_layers']
  len_node_labels = model_parameters['len_nodes']
  len_edge_labels = model_parameters['len_edges']
  node_label_dict = model_parameters['node_label_dict']
  edge_label_dict = model_parameters['edge_label_dict']
  node_label_dict = {value:key for key,value in node_label_dict.items()}
  edge_label_dict = {value:key for key,value in edge_label_dict.items()}
  node_rnn = CUSTOM_RNN_NODE(input_size=M, embedding_size=embedding_size_node_rnn,
                  hidden_size=hidden_size_node_rnn, number_layers=num_layers,output_size=hidden_size_edge_rnn,
              name="node",len_unique_node_labels=len_node_labels,len_unique_edge_labels=len_edge_labels)
  edge_rnn = CUSTOM_RNN_EDGE(input_size=1, embedding_size=embedding_size_edge_rnn,
                    hidden_size=hidden_size_edge_rnn, number_layers=num_layers, output_size=len_edge_labels,
                      name="edge",len_unique_edge_labels=len_edge_labels)


  fname_node = folder_param + "node_" + str(epoch) + ".dat"
  fname_edge= folder_param + "edge_" + str(epoch) + ".dat"
  node_rnn.load_state_dict(torch.load(fname_node))
  edge_rnn.load_state_dict(torch.load(fname_edge))


  num_graphs_to_be_generated = number_of_graphs
  max_num_nodes = model_parameters['max_num_nodes']
  M = model_parameters['M']
  num_layers = model_parameters['num_layers']
  most_frequent_edge_label = model_parameters['most_frequent_edge_label']
  node_label_freq_dict = model_parameters['node_label_freq_dict']
  node_rnn.hidden_n = node_rnn.init_hidden(num_graphs_to_be_generated)
  node_rnn.eval()
  edge_rnn.eval()
  generated_graphs =torch.zeros(num_graphs_to_be_generated, max_num_nodes-1, M)
  generated_graphs_labels = torch.zeros(num_graphs_to_be_generated,max_num_nodes-1,1)
  node_x = torch.ones(num_graphs_to_be_generated,1,M).long()*most_frequent_edge_label
  node_x_label = torch.ones(num_graphs_to_be_generated,1).long()
  for i in range(0,num_graphs_to_be_generated):
      node_x_label[i,0]=pick_random_label(node_label_freq_dict)
      #node_x_label[i,0] = 2
  node_x_label_1st_node = node_x_label

  print("generating")
  for i in range(0,max_num_nodes-1):
      print(i)
      h,h_mlp = node_rnn(node_x,node_x_label,None,is_packed=False,is_MLP=True)
      node_label_sampled = sample_multi(h_mlp,num_of_samples=1)
      h_edge_tmp = torch.zeros(num_layers-1, h.size(0), h.size(2))
      edge_rnn.hidden_n = torch.cat((h.permute(1,0,2),h_edge_tmp),dim=0)
      edge_x = torch.ones(num_graphs_to_be_generated,1,1).long()*most_frequent_edge_label
      
      
      node_x = torch.zeros(num_graphs_to_be_generated,1,M).long()
      node_x_label = node_label_sampled.long()
      
      
      for j in range(min(M,i+1)):
          edge_rnn_y_pred = edge_rnn(edge_x)
          edge_rnn_y_pred_sampled = sample_multi(edge_rnn_y_pred,num_of_samples=1)
         
          node_x[:,:,j:j+1] = edge_rnn_y_pred_sampled.view(edge_rnn_y_pred_sampled.size(0),
                                                          edge_rnn_y_pred_sampled.size(1),1)
          edge_x = edge_rnn_y_pred_sampled.long()
      
      
      
      generated_graphs_labels[:,i] = node_label_sampled
      generated_graphs[:, i:i + 1, :] = node_x

  predicted_graphs = []
  predicted_graphs_x = []
  predicted_graphs_x_labels = []
  for i in range(num_graphs_to_be_generated):
      pred_graph,pred_labels = cut_graph(generated_graphs[i].numpy(),generated_graphs_labels[i].numpy())
      predicted_graphs.append(decode_M_matrix(pred_graph,M))
      predicted_graphs_x.append(nx.from_numpy_matrix(predicted_graphs[i]))
      start_label = node_x_label_1st_node[i,0].tolist()
      pred_labels.insert(0,start_label)
      predicted_graphs_x_labels.append(pred_labels)
  max_num_edges_test = max([graph.number_of_edges() for graph in predicted_graphs_x])
  min_num_edges_test = min([graph.number_of_edges() for graph in predicted_graphs_x])
  max_num_nodes_test = max([graph.number_of_nodes() for graph in predicted_graphs_x])
  min_num_nodes_test = min([graph.number_of_nodes() for graph in predicted_graphs_x])
  mean_edges = np.mean([graph.number_of_edges() for graph in predicted_graphs_x])
  mean_nodes = np.mean([graph.number_of_nodes() for graph in predicted_graphs_x])
  print("Number of testing graphs, max nodes, min nodes , max edges , min edges , max prev nodes", max_num_nodes_test, min_num_nodes_test, max_num_edges_test, min_num_edges_test,M)
  print("mean nodes and edges,", mean_nodes , mean_edges)


  a,b = np.unique(generated_graphs_labels,return_counts=True)
  print("node label distribution ,", [i for i in zip(a,b)])
  tp = [graph.number_of_nodes() for graph in predicted_graphs_x]
  print(np.mean(tp),np.std(tp))

  tp_e= [graph.number_of_edges() for graph in predicted_graphs_x]
  print(np.mean(tp_e),np.std(tp_e))



  for i in range(0,len(predicted_graphs_x)):
      g = predicted_graphs_x[i]
      labels= predicted_graphs_x_labels[i]
      for j in range(0,len(labels)):
          g.node[j]["node_label"] = labels[j]
          
  save_graphs_nx(predicted_graphs_x,ofile,node_label_dict,edge_label_dict)
