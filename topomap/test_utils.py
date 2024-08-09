import os, errno
import networkx as nx
from .TopoMap import TopoMap
from .dataset_utils import fvecs_read, Index
from .utils import compute_mst_mlpack, compute_mst_ann
import numpy as np



def build_graph_mst(MST):
    '''
    Build nx graph from emst
    '''
    G = nx.Graph()
    n_nodes = len(MST) + 1
    G.add_nodes_from(range(n_nodes))

    edges = [0]*len(MST)
    for i in range(len(MST)):
        edges[i] = (int(MST[i,0]),int(MST[i,1]),MST[i,2])
    G.add_weighted_edges_from(edges)

    return G

def compute_difference_graphs(G1,G2):
    '''
    Compute statistics about two graphs MST graphs G1 and G2
    '''
    n_equal_edges = 0
    w_equal_edges = 0
    for edge in G1.edges.data("weight"):
        if(edge[1] in G2[edge[0]]):
            n_equal_edges+=1
            w_equal_edges += edge[2]
    return n_equal_edges, w_equal_edges

def compute_persitance_mst(mst):
    n_edges = len(mst)
    persistence = [(0,mst[i,2]) for i in range(n_edges)]
    return persistence


def plot_persitance_mst_and_dist(mst_ANN, mst_MLpack,dataset_name,):
    import gudhi
    import matplotlib.pyplot as plt
    max_weight_emst = np.max(mst_MLpack[:,2])
    max_weight_amst = np.max(mst_ANN[:,2])
    
    mst_MLpack[:,2] = mst_MLpack[:,2]/ max_weight_emst
    mst_ANN[:,2] = mst_ANN[:,2] / max_weight_amst
    
    I_mlpack = compute_persitance_mst(mst_MLpack)
    I_ANN = compute_persitance_mst(mst_ANN)

    dist = gudhi.bottleneck_distance(I_ANN, I_mlpack)


    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    gudhi.plot_persistence_diagram(I_mlpack,axes = axs[0])
    axs[0].set_title('Persistance - EMST')

    gudhi.plot_persistence_diagram(I_ANN, axes = axs[1])

    axs[1].set_title('Persistance - AMST')

    
    fig.suptitle(f'Persistence diagrams of MSTs - {dataset_name} dataset')
    fig.tight_layout()
    plt.show()
    

    return dist


def Compare_MSTs(EMST, ANN_MST,dataset_name):
    mst_mlpack_nx = build_graph_mst(EMST)
    mst_ANN_nx = build_graph_mst(ANN_MST)
    n_equal_edges, w_equal_edges =compute_difference_graphs(mst_ANN_nx,mst_mlpack_nx)
    n_edges = len(EMST)
    #ordering to avoid float point precision
    EMST = EMST[EMST[:, 2].argsort()]
    ANN_MST = ANN_MST[ANN_MST[:, 2].argsort()]
    total_weight = EMST[:,2].sum()
    approx_weight = ANN_MST[:,2].sum()
    percentual_weight_error = (approx_weight - total_weight) /total_weight 

    bn_dist = plot_persitance_mst_and_dist(ANN_MST, EMST,dataset_name)

    return {'Bottleneck distance':bn_dist,'# equal edges':n_equal_edges,'percentual equal edges':n_equal_edges/n_edges,'weight equal edges':w_equal_edges,'percentual weight of equal edges':w_equal_edges/total_weight, "Total weight error":percentual_weight_error}

def MST_test_topology(data_path, Index_path,dataset_name, save_folder,save_msts = False ):
    



    print("--")
    points  = fvecs_read(data_path)
    index = Index(Index_path)
    mst_ANN, ANN_building_time = compute_mst_ann(points,index, compute_time= True)
     
    mst_ANN_nx = build_graph_mst(mst_ANN)
    
    
    
    mst_mlpack, mlpack_building_time =  compute_mst_mlpack(points, compute_time = True)
    
    mst_mlpack_nx = build_graph_mst(mst_mlpack)

    bn_dist = plot_persitance_mst_and_dist(mst_ANN, mst_mlpack,dataset_name)
    n_equal_edges, w_equal_edges =compute_difference_graphs(mst_ANN_nx,mst_mlpack_nx)
    n_edges = len(mst_ANN)
    total_weight = mst_mlpack[:,2].sum()
    approx_weight = mst_ANN[:,2].sum()
    percentual_weight_error = abs(approx_weight/total_weight -1)

    if(save_msts):
        mst_filename = f'{save_folder}/mst_{dataset_name}.npy'
        vamana_mst_filename = f'{save_folder}/Vamana_mst_{dataset_name}.npy'
        # silentremove(mst_filename)
        # silentremove(vamana_mst_filename)
        np.save(mst_filename,mst_mlpack, allow_pickle=True)
        np.save(vamana_mst_filename,mst_ANN, allow_pickle= True)
        
    

    return [mlpack_building_time,ANN_building_time,bn_dist,n_equal_edges,n_equal_edges/n_edges,w_equal_edges,w_equal_edges/total_weight, percentual_weight_error]




        

