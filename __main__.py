from graph_partitions import *
from metrics import *



if __name__ == '__main__':
    
    partitionB,mojo_det,jaccard_det = ppr_partition(500)
    partitionA = ground_truth_partition()
    mojo = mojo_distance(partitionA,partitionB)
    jaccard = jaccard_similarity(partitionA, partitionB)
