from graph_partitions import *
from metrics import *
import matplotlib.pyplot as plt
import networkx

""" Load Graph """
def load_graph(path):
    
    # Load data
    dataset = pd.read_csv(path,header=None)
    
    #Convert to Series
    data = dataset.iloc[:,:][0]
    
    indexes = []
    
    # Keep only function dependecies
    for i in range(data.size):
        if data[i].count(')')==2:
            indexes.append(i)
            
    
    data = data.loc[indexes]
    data.index = range(data.size)
    
    G = networkx.Graph()
    
    # Create Graph
    for i in range(data.size):
        temp = data[i].split(";")
        G.add_node(temp[0])
        G.add_node(temp[1])
        G.add_edge(temp[0],temp[1])
        
    
    nodes = np.array(list(G.nodes()))
    #Nodes ids
    nodes_ids = [i + 1 for i in range(len(nodes))]
    
    #Extract communities names
    communities_names = dict()    
    for i in range(len(nodes)):
        temp = nodes[i].split('(')
        t = temp[0].split('.')
        com_name = t[-2]
        communities_names[com_name] = communities_names.get(com_name,0) + 1
    
    communities_names = list(communities_names.keys())
    
    #Communities ids
    communities_ids = [i + 1 for i in range(len(communities_names))]
    
    communities = {k: [] for k in communities_ids}
    
    # Fill communities
    for i in range(len(nodes)):
        temp = nodes[i].split('(')
        t = temp[0].split('.')
        com_name = t[-2]
        communities[communities_names.index(com_name)+1].append(i+1)
    
    
    # Relabel nodes
    G = networkx.convert_node_labels_to_integers(G,1)
    
    
    
    
    
    return G,communities



if __name__ == '__main__':
    
    path = "C:/Users/User/Desktop/Thesis/Dataset/log4j_1.0.4_features.csv"
    
    graph,communities = load_graph(path)
    
    node2community = {node: community for community, nodes in communities.items() for node in nodes}
    node2community = dict(sorted(node2community.items()))
    
    node2perturbed = perturb(node2community, graph, 400)
    method = "pagerank"
    
    node2remodularized = remodularize(method, graph, node2perturbed)
    
    for i in list(graph.nodes()):
        node2community[i]='A ' + str(node2community[i])
        node2remodularized[i]='B ' + str(node2remodularized[i])
        node2perturbed[i] = 'B ' + str(node2perturbed[i])
        
    
        
    
    mojo = mojo_distance(node2community,node2remodularized)
    mojo_det = mojo_distance(node2community,node2perturbed)
    
    jaccard = jaccard_similarity(node2community,node2remodularized)
    jaccard_det = jaccard_similarity(node2community,node2perturbed)
    
    
    
    
    
    
    
    
    
    
    
    
    