# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import networkx as nx
import numpy as np

from embedding_models import deep_infomax, low_dim_transformation
from graphs_module import draw_graph, graph_loader

from sklearn.neighbors import BallTree

import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

class graph_statistics:

  def __init__(self, graph_generator , orig_folder, synth_folder, dataset = 'citeseer',
   orig_graph_cnt = 13000, gen_graph_cnt = 3000, iter_count = 10, model_name = "epoch",
   another_load_method = False, batch_size = 100, node_label = 'label', edge_label = 'label', cut_label = False):
   
    if not another_load_method:
      orig_graph_nx = graph_loader.load(orig_folder, batch_size)

    else:
      orig_graph_nx = graph_loader.load_pickle(orig_folder, orig_graph_cnt, batch_size, True, dataset, node_label, edge_label, cut_label)

    degree_centralities = list()
    harmonic_centralities = list()
    beetweenness_centralities = list()
    component_sizes = list()

    self.transformed_embs = list()

    degree_centralities.append(nx.degree_centrality(orig_graph_nx))
    harmonic_centralities.append(nx.harmonic_centrality(orig_graph_nx))
    beetweenness_centralities.append(nx.betweenness_centrality(orig_graph_nx))
    component_sizes.append({0 : [len(c) for c in nx.connected_components(orig_graph_nx)]})


    if not another_load_method:
      for i in range(1 , iter_count + 1):
        graph_filename = synth_folder + "/graph_" + str(i)
        g_graph_nx, g_graph, g_features_df, g_embedding, g_emb_transformed = graph_generator(graph_filename, trained_model_name = model_name)
        degree_centralities.append(nx.degree_centrality(g_graph_nx))
        harmonic_centralities.append(nx.harmonic_centrality(g_graph_nx))
        beetweenness_centralities.append(nx.betweenness_centrality(g_graph_nx))
        component_sizes.append({i : [len(c) for c in nx.connected_components(g_graph_nx)]})

    else:
      for i in range(1 , iter_count + 1):
        g_graph_nx = graph_loader.load_pickle(synth_folder, gen_graph_cnt, batch_size, True, dataset, node_label, edge_label, cut_label )
        degree_centralities.append(nx.degree_centrality(g_graph_nx))
        harmonic_centralities.append(nx.harmonic_centrality(g_graph_nx))
        beetweenness_centralities.append(nx.betweenness_centrality(g_graph_nx))
        component_sizes.append({i : [len(c) for c in nx.connected_components(g_graph_nx)]})


    self.degree_centralities = pd.DataFrame(degree_centralities).swapaxes("index", "columns")
    self.harmonic_centralities = pd.DataFrame(harmonic_centralities).swapaxes("index", "columns")
    self.beetweenness_centralities = pd.DataFrame(beetweenness_centralities).swapaxes("index", "columns")
    self.component_sizes = pd.DataFrame(component_sizes).swapaxes("index", "columns")

    self.component_sizes = component_sizes[0]
    for i in range(1,iter_count + 1):
      self.component_sizes.update(component_sizes[i])

    for i in self.component_sizes:
      print(len(self.component_sizes[i]))
    self.component_sizes = pd.DataFrame(self.component_sizes)

 

  def draw_hist(self, df, name):
    sns.set(style='whitegrid')
    sns.set(font_scale = 2)
    fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(25,15))

    l1 = sns.histplot(data=df[1], ax = axs[0,0], kde = True).set_title('Synthetic\n graph 1', fontsize= 16)
    l2 = sns.histplot(data=df[2], ax = axs[0,1], kde = True).set_title('Synthetic\n graph 2', fontsize= 16)
    l3 = sns.histplot(data=df[3], ax = axs[0,2], kde = True).set_title('Synthetic\n graph 3', fontsize= 16)
    l4 = sns.histplot(data=df[4], ax = axs[1,0], kde = True).set_title('Synthetic\n graph 4', fontsize= 16)
    l5 = sns.histplot(data=df[5], ax = axs[1,1], kde = True).set_title('Synthetic\n graph 5', fontsize= 16)
    l6 = sns.histplot(data=df[6], ax = axs[1,2], kde = True).set_title('Synthetic\n graph 6', fontsize= 16)
    l7 = sns.histplot(data=df[7], ax = axs[2,0], kde = True).set_title('Synthetic\n graph 7', fontsize= 16)
    l8 = sns.histplot(data=df[8], ax = axs[2,1], kde = True).set_title('Synthetic\n graph 8', fontsize= 16)
    l9 = sns.histplot(data=df[9], ax = axs[2,2], kde = True).set_title('Synthetic\n graph 9', fontsize= 16)
    l10 = sns.histplot(data=df[10], ax = axs[3,0], kde = True ).set_title('Synthetic\n graph 10', fontsize= 16)
    lorig = sns.histplot(data=df[0], ax = axs[3,1], color = "red", kde = True).set_title('Original\ngraph', fontsize= 16)
    axs[3,1].set(xlabel=None)
    axs[3,1].tick_params(labelrotation=45)
    axs[3,0].set(xlabel=None)
    axs[3,0].tick_params(labelrotation=45)
    

    fig.delaxes(axs[3,2])


    fig.suptitle(name + " centrality histograms", fontsize=22)
    sns.despine(top=True,
                right=True,
                left=True,
                bottom=False)
    plt.subplots_adjust(right=0.9)

  def draw_boxplot(self, df, name):
    sns.set(style='whitegrid')
    
    fig, ax = plt.subplots(figsize=(20,6))

    g = sns.boxplot(data=df, width=0.7,notch=True, showcaps=False,
        flierprops={"marker": "o"},
     
        medianprops={"color": "coral"})
    plt.title(name + " centrality boxplots", fontsize=16)
    sns.despine(top=True,
                right=True,
                left=True,
                bottom=False)
    
    xvalues = ["Original\ngraph"] + ["synthetic graph\n" + str(i) for i  in range(1, len(df.columns))]

 
    plt.xticks(np.arange(len(df.columns)), xvalues)
    plt.tight_layout()


  def degree_centrality_hist(self):
    self.draw_hist(self.degree_centralities, "Degree")


  def degree_centrality_boxplot(self):
    self.draw_boxplot(self.degree_centralities, "Degree")
    

  def harmonic_centrality_hist(self):
    self.draw_hist(self.harmonic_centralities, "Harmonic")

  def harmonic_centrality_boxplot(self):
    self.draw_boxplot(self.harmonic_centralities, "Harmonic")

  def beetweenness_centrality_hist(self):
    self.draw_hist(self.beetweenness_centralities, "Beetweeness")


  def beetweenness_centrality_boxplot(self):
    self.draw_boxplot(self.beetweenness_centralities, "Beetweeness")



  def components_sizes_hist(self):
    
    df = self.component_sizes
    sns.set(style='whitegrid')
    sns.set(font_scale = 2)
    fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(25,15))

    l1 = sns.histplot(data=df[1], ax = axs[0,0], kde = True).set_title('Synthetic\n graph 1', fontsize= 16)
    l2 = sns.histplot(data=df[2], ax = axs[0,1], kde = True).set_title('Synthetic\n graph 2', fontsize= 16)
    l3 = sns.histplot(data=df[3], ax = axs[0,2], kde = True).set_title('Synthetic\n graph 3', fontsize= 16)
    l4 = sns.histplot(data=df[4], ax = axs[1,0], kde = True).set_title('Synthetic\n graph 4', fontsize= 16)
    l5 = sns.histplot(data=df[5], ax = axs[1,1], kde = True).set_title('Synthetic\n graph 5', fontsize= 16)
    l6 = sns.histplot(data=df[6], ax = axs[1,2], kde = True).set_title('Synthetic\n graph 6', fontsize= 16)
    l7 = sns.histplot(data=df[7], ax = axs[2,0], kde = True).set_title('Synthetic\n graph 7', fontsize= 16)
    l8 = sns.histplot(data=df[8], ax = axs[2,1], kde = True).set_title('Synthetic\n graph 8', fontsize= 16)
    l9 = sns.histplot(data=df[9], ax = axs[2,2], kde = True).set_title('Synthetic\n graph 9', fontsize= 16)
    l10 = sns.histplot(data=df[10], ax = axs[3,0], kde = True).set_title('Synthetic\n graph 10', fontsize= 16)
    lorig = sns.histplot(data=df[0], ax = axs[3,1], color = "red", kde = True).set_title('Original\ngraph', fontsize= 16)
    axs[3,1].set(xlabel=None)
    axs[3,0].set(xlabel=None)
    fig.delaxes(axs[3,2])
    
    fig.suptitle("Components size histograms", fontsize=22)
    sns.despine(top=True,
                right=True,
                left=True,
                bottom=False)
    plt.subplots_adjust(right=0.9)




    
