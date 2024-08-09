import numpy as np
from scipy.sparse import csr_matrix
import os
def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))



def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def read_and_convert(file,nbits):
    '''Receive the file and nbits and return the integer that is encoded.
    nbits = 8 or 4'''
    encoded_int = file.read(nbits)
    encoded_int = int.from_bytes(encoded_int,byteorder='little',signed = False)
    return encoded_int


def Index_read(filename, verbose = False):
        #I'm not using the start since we won't search, but if necessary it can be returned from this function
        file = open(filename, "rb")

        expected_file_size = read_and_convert(file,8)
        max_observed_degree = read_and_convert(file,4)
        start = read_and_convert(file,4)
        file_fronzen_pts = read_and_convert(file,8)       
        bytes_read = 8 + 4 + 4 + 8 
        nodes_read = 0
        #I don't know the length of the graph in advance, so I'm using a dict instead of appending several times in a list of lists
        adj_dict ={}
        cc =0
        while( bytes_read != expected_file_size):  
            number_of_neighbors = read_and_convert(file,4)
            cc+=number_of_neighbors
            if( number_of_neighbors == 0):
                print(f"ERROR: Point Found with no out-neighbours, point# {nodes_read} \n")        
            buffer = file.read(4* number_of_neighbors)
            ngb = np.frombuffer(buffer, dtype=np.int32)
            adj_dict[nodes_read] = ngb
            nodes_read+=1
            bytes_read += 4*(number_of_neighbors + 1)
        file.close()
        if(verbose):
            print(f'From graph header, expected file size:{expected_file_size}\n max_observed_degree:{max_observed_degree} \n file_frozen_pts: {file_fronzen_pts}\n ')
            print(f'done. Index has {nodes_read} nodes')
        return adj_dict

def turn_symmetric(adj_dict):
    '''
    Assumes that adj_dict is a dictionary that has each adjacency_list associated with each key
    Returns a adj_dict where 
    '''

    for k in adj_dict.keys():
        for ngb in adj_dict[k]:
            if k not in adj_dict[ngb]:
                adj_dict[ngb] = np.append(adj_dict[ngb],k)

    return adj_dict

def Compute_distance(dataset,i,j, aproximate_almost_zero = False,seed = None):
    from random import random
    distance = np.linalg.norm(dataset[i,:]- dataset[j,:])
    if (aproximate_almost_zero):
        if(seed):
            random.seed(seed)
        random_noise = 1 +random()
        distance = distance + 1e-7*random_noise
    return distance

def Compute_adj_matrix(adj_dict, input = None, input_file ="", turn_sym = False, aproximate_almost_zero = False, ):
    '''
    Compute adjacency matrix using sparse scipy representation. 
    Assumes that adj_dict is a dictionary that has each adjacency_list associated with each key
    input_file is the path for the input_file. Assumed to be on fvecs format for a while
    '''
    if(input_file):
        input = fvecs_read(input_file)
    number_of_nodes = len(adj_dict)

    
    if(turn_sym):
        adj_dict = turn_symmetric(adj_dict)
    total_edges = 0
    for v in adj_dict.values():
        total_edges += len(v)
    distances = np.zeros(total_edges)
    indptr = np.zeros(number_of_nodes+1)
    edges_read = 0
    indices = np.zeros(total_edges)
    for i in range(number_of_nodes):
        neighbors = adj_dict[i]
        for neighbor in neighbors:
            distances[edges_read] = Compute_distance(input,i,neighbor,aproximate_almost_zero)
            indices[edges_read] = neighbor
            edges_read+=1
        
        indptr[i+1] = indptr[i]+len(neighbors)
    
    
    sparse_matrix = csr_matrix((distances,indices,indptr),shape=(number_of_nodes,number_of_nodes))
        
    return sparse_matrix
     


def Compute_MST_from_adj(adj_matrix):
    from scipy.sparse.csgraph import  minimum_spanning_tree
    from scipy.sparse import issparse
    is_sparse = issparse(adj_matrix)
    if(is_sparse):
        adj_matrix.data += 1
        MST = minimum_spanning_tree(adj_matrix)
        MST.data -= 1
    else:
        MST = minimum_spanning_tree(adj_matrix)
    return MST


class Index:
    """Read Index generated using DISKANN microsoft implementation
    """
    def __init__(self, index_path) -> None:
        self.index_path = index_path

    def get_mst(self, points):
        Index = Index_read(self.index_path)
        sparse_adj_matrix = Compute_adj_matrix(Index, input = points)
        mst = Compute_MST_from_adj(sparse_adj_matrix)
        return mst

def silentremove(filename):
    '''
    Delete a file if it exists without raise an error.  
    '''
    try:
        os.remove(filename)
    except: 
        pass 



    

    
