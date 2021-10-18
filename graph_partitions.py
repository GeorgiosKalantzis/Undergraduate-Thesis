import networkx
import pandas as pd
import numpy as np
import random
import community as community_louvain
import pygrank as pg
from pygrank.algorithms import HeatKernel
from pygrank.algorithms import AbsorbingWalks
from metrics import *
""" Load Graph """
def load_graph():
    
    # Load data
    dataset = pd.read_csv("C:/Users/User/Desktop/Thesis/Dataset/squirrel-sql_1.0_features.csv",header=None)
    
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
    
    return G

""" Change the communities and deteriorate the organization """
def deterioration_proccess(num_of_edges):
    G = load_graph()
    nodes_list = np.array(list(G.nodes()))
    edges_list = np.array(list(G.edges()))
    
    # New nodes_list with changed communities to be returned
    new_nodes_list = nodes_list.copy()
    for i in range(10):
        np.random.shuffle(edges_list)
        c = 0
        prev1 = 0
        prev2 = 0
        for i in range(len(edges_list)):
            
            communities = dict()       
            for j in range(len(new_nodes_list)):
                temp = new_nodes_list[j].split('(')
                t = temp[0].split('.')
                com_name = t[-2]
                communities[com_name] = communities.get(com_name,0) + 1
                
            
            c = c + 1
            if(c == num_of_edges):
                break
            
            index1 = np.where(edges_list[i][0] == nodes_list)
            index2 = np.where(edges_list[i][1] == nodes_list)
            
            if(nodes_list[index1[0][0]] != new_nodes_list[index1[0][0]] or nodes_list[index2[0][0]] != new_nodes_list[index2[0][0]]):
                continue
            
            temp1 = edges_list[i][0].split('(')
            t1 = temp1[0].split('.')
            com_name1 = t1[-2]
            
            temp2 = edges_list[i][1].split('(')
            t2 = temp2[0].split('.')
            com_name2 = t2[-2]
            
            
            
            if(communities[com_name1] == 1 and communities[com_name2] == 1):
                continue
            if(communities[com_name1] == 1):
                com_name2 = com_name1
                t1[-2] = com_name1
                t2[-2] = com_name2
                
                strt1 = '.'.join(t1)
                strt2 = '.'.join(t2)
                
                l1 = []
                l1.append(strt1)
                l1.append(temp1[1])
                
                l2 = []
                l2.append(strt2)
                l2.append(temp2[1])
                
                new_nodes_list[index1[0][0]] = '('.join(l1)
                new_nodes_list[index2[0][0]] = '('.join(l2)
                continue
            
            if(communities[com_name2] == 1):
                com_name1 = com_name2
                t1[-2] = com_name1
                t2[-2] = com_name2
                
                strt1 = '.'.join(t1)
                strt2 = '.'.join(t2)
                
                l1 = []
                l1.append(strt1)
                l1.append(temp1[1])
                
                l2 = []
                l2.append(strt2)
                l2.append(temp2[1])
                
                new_nodes_list[index1[0][0]] = '('.join(l1)
                new_nodes_list[index2[0][0]] = '('.join(l2)
    
                continue
                    
            
            
            
            if (com_name1 != com_name2):
                p = random.random()
                if p >= 0.5:
                    com_name1 = com_name2
                else:
                    com_name2 = com_name1
            """   
            if i >=1:
                if (index1 == prev1):
                    com_name2 = com_name1
                if(index2 == prev2):
                    com_name1 = com_name2
                if(index1 == prev2):
                    com_name1 = com_name2
                if(index2 == prev1):
                    com_name2 = com_name1
            """    
        
            t1[-2] = com_name1
            t2[-2] = com_name2
            
            strt1 = '.'.join(t1)
            strt2 = '.'.join(t2)
            
            l1 = []
            l1.append(strt1)
            l1.append(temp1[1])
            
            l2 = []
            l2.append(strt2)
            l2.append(temp2[1])
            
            new_nodes_list[index1[0][0]] = '('.join(l1)
            new_nodes_list[index2[0][0]] = '('.join(l2)
            
        
            prev1 = index1
            prev2 = index2
        
    
    return new_nodes_list
    
    


""" Page rank on the deteriorated community structure """
def ppr_partition(num_of_edges):
    
    G = load_graph()        
        
    nodes_list = np.array(list(G.nodes()))
    new_nodes_list = deterioration_proccess(num_of_edges)
    
    
    personalization_array = []
    personalization = {k: 0 for k in nodes_list}
    
    """ Partition of deteriorated Graph to study improvement """
    mylist = [i+1 for i in range(len(nodes_list))]
    original_partition = {k : '' for k in mylist}
    
    
    expected_communities = dict()       
    for i in range(len(new_nodes_list)):
        temp = new_nodes_list[i].split('(')
        t = temp[0].split('.')
        com_name = t[-2]
        expected_communities[com_name] = expected_communities.get(com_name,0) + 1    
    
    coms_name = list(expected_communities.keys()) 
    
    for i in range(len(new_nodes_list)):
        temp = new_nodes_list[i].split('(')
        t = temp[0].split('.')
        com_name = t[-2]
        original_partition[i+1] = 'B ' + str(coms_name.index(com_name) + 1)
    
    partitionA = ground_truth_partition()
    
    mojo_deteriorated = mojo_distance(partitionA,original_partition)
    jaccard_deteriorated = jaccard_similarity(partitionA,original_partition)
    
    """    
    Page rank
    """

    communities = dict()       
    for i in range(len(nodes_list)):
        temp = nodes_list[i].split('(')
        t = temp[0].split('.')
        com_name = t[-2]
        communities[com_name] = communities.get(com_name,0) + 1    
    
    coms_name = list(communities.keys()) 
    
    for i in coms_name:
        
        personalization = {k: 0 for k in nodes_list}
        for j in range(len(new_nodes_list)):
            temp = new_nodes_list[j].split('(')
            t = temp[0].split('.')
            com_name = t[-2]
            if (i == com_name):
                personalization[nodes_list[j]] = 1
        
        personalization_array.append(personalization)
        
    
    mylist = [i+1 for i in range(len(nodes_list))]
    ppr_partition = {k : '' for k in mylist}
    ppr_coms = {k : [] for k in nodes_list}
    
    #ranker = pg.PageRank(alpha=0.9, tol=1.E-9, max_iters=1000, normalization="symmetric")
    ranker = AbsorbingWalks(0.9,tol=1.E-9,max_iters=1000)
    
    
    for i in range(len(personalization_array)):
        ranks = ranker(G,personalization_array[i])
        for j in range(len(nodes_list)):
            ppr_coms[nodes_list[j]].append(ranks[nodes_list[j]])
    """    
    for i in range(len(personalization_array)):
        ppr = networkx.pagerank(G, alpha=0.6, personalization=personalization_array[i],tol=1e-09,max_iter=1000)
        for j in list(ppr.keys()):
            ppr_coms[j].append(ppr[j])
    """ 
    coms = []
    v = list(communities.values())
    for i in list(ppr_coms.keys()):
        index = np.where(nodes_list == i)
        c = sorted(range(len(ppr_coms[i])), key=lambda k:ppr_coms[i][k])
        coms.append(c[-1])
    
    """
    final_coms = list(set(coms))
    personalization_array = []
    for i in final_coms:
        personalization = {k: 0 for k in nodes_list}
        for j in range(len(coms)):
            if (coms[j]==i):
                personalization[nodes_list[j]] = 1
        personalization_array.append(personalization)
    
    ppr_coms = {k : [] for k in nodes_list}
    for i in range(len(personalization_array)):
        ranks = ranker(G,personalization_array[i])
        for j in range(len(nodes_list)):
            ppr_coms[nodes_list[j]].append(ranks[nodes_list[j]])
        
        
    coms = []
    v = list(communities.values())
    for i in list(ppr_coms.keys()):
        index = np.where(nodes_list == i)
        c = sorted(range(len(ppr_coms[i])), key=lambda k:ppr_coms[i][k])
        coms.append(c[-1])
    """   
    final_coms = list(set(coms))       
        
        
    
    for i in range(len(coms)):
        coms[i] = final_coms.index(coms[i]) + 1
    
    for i in range(len(coms)):
        ppr_partition[i+1] = 'B ' + str(coms[i])

    
    return ppr_partition,mojo_deteriorated,jaccard_deteriorated

""" Greedy modularity maximization on the deteriorated community structure """

def modularity_partition(num_of_edges):
    
    G = load_graph()
    nodes_list = np.array(list(G.nodes()))
    new_nodes_list = deterioration_proccess(num_of_edges)
  
    """ Partition of deteriorated Graph to study improvement """
    mylist = [i+1 for i in range(len(nodes_list))]
    original_partition = {k : '' for k in mylist}
    
    
    expected_communities = dict()       
    for i in range(len(new_nodes_list)):
        temp = new_nodes_list[i].split('(')
        t = temp[0].split('.')
        com_name = t[-2]
        expected_communities[com_name] = expected_communities.get(com_name,0) + 1    
    
    coms_name = list(expected_communities.keys()) 
    
    for i in range(len(new_nodes_list)):
        temp = new_nodes_list[i].split('(')
        t = temp[0].split('.')
        com_name = t[-2]
        original_partition[i+1] = 'B ' + str(coms_name.index(com_name) + 1)
        
    """
    Modularity
    """
    
    mapping = {nodes_list[i]: i+1 for i in range(len(nodes_list))}
    
    G = networkx.relabel_nodes(G,mapping)
    for i in original_partition.keys():
        original_partition[i]=[int(s) for s in original_partition[i].split() if s.isdigit()][0]
        
    partition = community_louvain.best_partition(G,original_partition)
    for i in partition.keys():
        p = partition[i]+1
        partition[i]='B ' + str(p)
    
    return partition
 
    

""" Origina partition of the graph -> Ground truth """

def ground_truth_partition():
        
    G = load_graph()
        
    nodes_list = np.array(list(G.nodes()))
    mylist = [i+1 for i in range(len(nodes_list))]
    
    # --- Extract partition ----
    
    expected_communities = dict()       
    for i in range(len(nodes_list)):
        temp = nodes_list[i].split('(')
        t = temp[0].split('.')
        com_name = t[-2]
        expected_communities[com_name] = expected_communities.get(com_name,0) + 1    

    coms_name = list(expected_communities.keys())
        
    original_partition = {k : '' for k in mylist}

    for i in range(len(nodes_list)):
        temp = nodes_list[i].split('(')
        t = temp[0].split('.')
        com_name = t[-2]
        original_partition[i+1] = 'A ' + str(coms_name.index(com_name) + 1)
        
    return original_partition

""" Deteriorated partition produced -> Compare with above algorithms to see improvement """

def deteriorated_partition(num_of_edges):
    
    G = load_graph()
        
        
    nodes_list = np.array(list(G.nodes()))
    new_nodes_list = deterioration_proccess(num_of_edges)
    
    """ Partition of deteriorated Graph to study improvement """
    mylist = [i+1 for i in range(len(nodes_list))]
    original_partition = {k : '' for k in mylist}
    
    
    expected_communities = dict()       
    for i in range(len(new_nodes_list)):
        temp = new_nodes_list[i].split('(')
        t = temp[0].split('.')
        com_name = t[-2]
        expected_communities[com_name] = expected_communities.get(com_name,0) + 1    
    
    coms_name = list(expected_communities.keys()) 
    
    for i in range(len(new_nodes_list)):
        temp = new_nodes_list[i].split('(')
        t = temp[0].split('.')
        com_name = t[-2]
        original_partition[i+1] = 'B ' + str(coms_name.index(com_name) + 1)
        
    return original_partition
    
    
                
    
    

    
    