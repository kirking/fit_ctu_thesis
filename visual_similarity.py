# -*- coding: utf-8 -*-
"""visual_similarity.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kWQU5gmqzSM03hsvPBmSSBX3i8mhDT64
"""

# Commented out IPython magic to ensure Python compatibility.
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity, paired_euclidean_distances, paired_manhattan_distances

from graphs_module import draw_graph
from graphs_module import graph_loader

from graph_l_rnn_module import graph_label_rnn_generate

from embedding_models import deep_infomax, g_sage, low_dim_transformation

from embedding_models import node2Vec_embedding, riwalk_embedding

import pandas as pd
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



# %matplotlib inline

class vis_sim:

  def emb_graph_to_vec(self, emb, scale = True):


    if scale:
      min_max_scaler = preprocessing.MinMaxScaler()
      emb = min_max_scaler.fit_transform(emb)
    df = pd.DataFrame(emb)
    return np.array(df.mean(axis=0))

  def __init__(self, orig_filename = "data/citation.txt", graph_filename  = "generate_experiment/graphs/graph", trained_model_name = "epoch", emb_model = "node2vec", scale = True, dataset = None, load_synt = False, orig_g_count = 13000, synt_g_count = 3000, batch_size = 100, label_name = 'node_label', ed_label = 'weight', cut_label = False):

    self.edge_label = ed_label
    if load_synt:
      
      orig_graph_nx = graph_loader.load_pickle(orig_filename, orig_g_count, batch_size, True, dataset, label_name, ed_label, cut_label)
      #load/generate synt graph
      g_graph_nx = graph_loader.load_pickle(graph_filename, synt_g_count, batch_size, True, dataset, label_name,ed_label, cut_label)
      

    else:
      #load orig graph
      orig_graph_nx = graph_loader.load(orig_filename, batch_size, random_sampling = True)
      #load/generate synt graph
      graph_label_rnn_generate(graph_filename + str("_vis_sim_test"), name = trained_model_name)
    
      g_graph_nx = graph_loader.load(graph_filename+ str("_vis_sim_test"), batch_size)
      

    #embedding
  
    self.orig_graphs_embs = list()

    self.synth_graphs_embs = list()

    self.orig_graphs = [orig_graph_nx.subgraph(c).copy() for c in nx.connected_components(orig_graph_nx) if len(c) > 1]

    self.synth_graphs = [g_graph_nx.subgraph(c).copy() for c in nx.connected_components(g_graph_nx) if len(c) > 1]


    for g in self.orig_graphs:
      if emb_model == "node2vec":
        if load_synt:
          g_sg, g_labels= graph_loader.transform_to_sg(g, 'topic')
        else:
          g_sg, g_labels = graph_loader.transform_to_sg(g)
        
        g_emb = node2Vec_embedding(g_sg, None)

      else:
        g_emb = riwalk_embedding(g, None)
      if len(g_emb) > 1: 
        self.orig_graphs_embs.append(self.emb_graph_to_vec(g_emb, scale))

    for g in self.synth_graphs:
      if emb_model == "node2vec":
        if load_synt:
          g_sg, g_labels = graph_loader.transform_to_sg(g, 'topic')
        else:
          g_sg, g_labels = graph_loader.transform_to_sg(g)
        
          g_emb = node2Vec_embedding(g_sg, None)

      else:
        g_emb = riwalk_embedding(g, None)

      if len(g_emb) > 1:
        self.synth_graphs_embs.append(self.emb_graph_to_vec(g_emb, scale))
        
  def measure_similarity(self):
    self.results_cos = list()
    for or_indx,orig in enumerate(self.orig_graphs_embs):
      for g_indx,gen in enumerate(self.synth_graphs_embs):
        self.results_cos.append([cosine_similarity(gen.reshape(1,-1), orig.reshape(1,-1)), or_indx, g_indx])

    self.results_euc = list()
    for or_indx,orig in enumerate(self.orig_graphs_embs):
      for g_indx,gen in enumerate(self.synth_graphs_embs):
        self.results_euc.append([paired_euclidean_distances(gen.reshape(1,-1), orig.reshape(1,-1)), or_indx, g_indx])

    self.results_man = list()
    for or_indx,orig in enumerate(self.orig_graphs_embs):
      for g_indx,gen in enumerate(self.synth_graphs_embs):
        self.results_man.append([paired_manhattan_distances(gen.reshape(1,-1), orig.reshape(1,-1)), or_indx, g_indx])

    self.results_cos.sort(reverse= True, key = lambda t: t[0])

    self.results_euc.sort(reverse= False,key = lambda t: t[0])

    self.results_man.sort(reverse= False,key = lambda t: t[0])

  def draw_sim_graphs(self,sim_indx = 0, orig_dir = 'vis_sim/orig.png', synt_dir = 'vis_sim/synth.png', metric = "euc"):

    if metric == "cos":
      res = self.results_cos
    elif metric == "man":
      res = self.results_man
    else:
      res = self.results_euc

    orig_sim = self.orig_graphs[res[sim_indx][1]]
    g_sim =  self.synth_graphs[res[sim_indx][2]]

    draw_graph(orig_sim, orig_dir, 'topic', self.edge_label)

    draw_graph(g_sim, synt_dir, 'topic', self.edge_label)

    return orig_sim, g_sim