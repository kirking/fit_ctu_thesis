import networkx as nx
import random
def read_graphs_in_networkx(infile,labels=True, num_graphs_to_read=10000000, random_sampling = False):
    data = [s for s in open(infile,"r").read().splitlines() if s]
    index = 0 
    graphs = []
    node_label_dict = {}
    node_label_ct = 1
    edge_label_dict = {}
    edge_label_ct = 1
    node_label_freq_dict = {}
    edge_label_freq_dict= {} 

    # main loop
    while index < len(data):
        item = data[index]
        index = index+1
        
        if index < len(data) and len(item) > 0 and  item[0] == '#' and len(graphs) < num_graphs_to_read:
            if random_sampling:
              rnd = random.randint(0,11)
          #43
              for i in range(0,rnd):
                num_nodes = int(data[index])
                index = index+1 #2
                index = index + num_nodes
                num_edges = int(data[index])
                index = index+1
                index = index + num_edges + 1

            g = nx.Graph()
            num_nodes = int(data[index])
            index = index+1
            g.add_nodes_from(range(0,num_nodes))
            for i in range(num_nodes):
                node_label = data[index]

                index = index+1
                if labels:
                    if node_label not in node_label_dict:
                        node_label_dict[node_label] = node_label_ct
                        node_label_freq_dict[node_label_ct] = 0

                        node_label_ct = node_label_ct +1

                    node_label = node_label_dict[node_label]
                    node_label_freq_dict[node_label] +=1

                    g.node[i]["node_label"] = node_label
            

            num_edges = int(data[index])

            index = index+1
            edges = []
            edges_wt = []
            for i in range(num_edges):
                tmp = data[index].split()

                index+=1

                if len(tmp) == 3:    ### contains start node , end node and label
                    [start_node,end_node,edge_label] = tmp
                else:
                    [start_node,end_node] = tmp
                    edge_label = "dummy"  ### dummy var
                start_node = int(start_node)
                end_node = int(end_node)
                
                if edge_label not in edge_label_dict:
                    edge_label_dict[edge_label] = edge_label_ct
                    edge_label_freq_dict[edge_label_ct] = 0
                    edge_label_ct+= 1
                    
                edge_label = edge_label_dict[edge_label]
                edge_label_freq_dict[edge_label] +=1
                if labels:
                    edges.append((start_node,end_node,{'edge_label':edge_label}))
                    edges_wt.append((start_node,end_node,edge_label))
                else:
                    edges.append((start_node,end_node))
                    

            g.add_weighted_edges_from(edges_wt)
            graphs.append(g)
            
    return (graphs,node_label_dict,edge_label_dict,node_label_freq_dict,edge_label_freq_dict)
        


def save_graphs_nx(graphs,fname,node_label_dict,edge_label_dict):
    f = open(fname,"w")
    ct = 0
    for graph  in graphs:
        f.write("#"+str(ct)+"\n")
        f.write(str(graph.number_of_nodes()) +"\n")
        for i in range(0,graph.number_of_nodes()):
            f.write(str(node_label_dict[int(graph.node[i]["node_label"])])+"\n")
            
        f.write(str(graph.number_of_edges())+"\n")
        for edge in graph.edges:
            lb = str(edge_label_dict[int(graph[edge[0]][edge[1]]['weight'])])
            f.write(str(edge[0]) + " "+str(edge[1]) + " " +lb +"\n")
        f.write("\n")
        ct += 1
    f.close()

            
        
