import numpy as np 
from numpy.linalg import norm
from scipy.spatial import ConvexHull
from pathlib import Path
import shutil
import os
import mlpack
import diskannpy as dap

from .dataset_utils import Index, silentremove

class AuxHull():
    def __init__(self, points, simplices):
        self.points = points
        self.simplices = simplices
        self.vertices = np.array(list(range(len(points))))

def get_hull(points:np.ndarray, aligned_points = False):
    if len(points) == 1:
        hull = AuxHull(points=points,
                       simplices=np.array([[0,0]]))
    elif len(points) == 2:
        hull = AuxHull(points=points,
                       simplices=np.array([[0,1], [1,0]]))
    elif aligned_points:
        simplices = [[i,(i+1)%len(points)] for i in range(len(points))]
        hull = AuxHull(points = points,
                       simplices = np.array(simplices))
    else:
        hull = ConvexHull(points)
        
    return hull

def distance_segment_point(segment, p):
    a, b = segment
    
    if a[0]==b[0] and a[1]==b[1]:
        return norm(a-p)
    if (a[0]==p[0] and a[1]==p[1]) or (p[0]==b[0] and p[1]==b[1]):
        return 0
    else:
        proj_p_ab = np.dot((p-a), (b-a))
        norm_proj_p_ab = proj_p_ab / (norm(b-a)**2)

        if norm_proj_p_ab <= 0:
            closest_point = a
        elif norm_proj_p_ab>= 1:
            closest_point = b
        else:
            closest_point = a + norm_proj_p_ab*(b-a)

        return norm(closest_point-p)

def closest_edge_point(hull, 
                       ref_point:np.ndarray):
    '''
    Find edge of convex hull that is closer to the reference point.
    Returns the points of the edge and its indexes.
    '''
    closest_i = -1
    closest_dist = np.inf

    for i in range(len(hull.simplices)):
        a, b = hull.points[hull.simplices[i]]
        d = distance_segment_point((a, b), ref_point)
        
        if d < closest_dist:
            closest_dist = d
            closest_i = i

    closest_edge = hull.points[hull.simplices[closest_i]]

    return closest_edge, hull.simplices[closest_i]

def closest_edge_point_simplices(component_simplices, 
                                 projections,
                                 ref_point:np.ndarray):
    '''
    Find edge of convex hull that is closer to the reference point.
    Returns the points of the edge and its indexes.
    '''
    closest_i = -1
    closest_dist = np.inf

    for i in range(len(component_simplices)):
        a, b = projections[component_simplices[i],:]
        d = distance_segment_point((a, b), ref_point)
        
        if d < closest_dist:
            closest_dist = d
            closest_i = i

    closest_simplice = component_simplices[closest_i]
    closest_edge = projections[closest_simplice,:]

    return closest_edge, closest_simplice

def find_angle(segment:np.ndarray):
    '''
    Find angle to rotate component so that segment is paralel to the vertical axis
    '''
    p1, p2 = segment
    
    if p1[1] == p2[1]:
        # Angle is 0 degree
        return 1, 0
    if p1[0] == p2[0]:
        # Angle is 90 degree
        return 0, 1

    # Use rightmost vertice as reference for angle
    if p1[0] < p2[0]:
        vec = p2 - p1
    else:
        vec = p1 - p2

    hip = norm(vec)
    vec[0] /= hip
    vec[1] /= hip

    cos = vec[0]
    sin = np.sqrt(1 - cos**2)
    if vec[1] >= 0:
        sin = -1*sin

    return cos, sin

class Transform():
    def __init__(self, x=0, y=0, sin=0, cos=1, scalar=1) -> None:
        self.x = x
        self.y = y
        self.sin = sin
        self.cos = cos
        self.scalar = scalar

    def translate(self, points:np.ndarray) -> np.ndarray:
        trans_points = np.zeros(shape=(len(points),2))
        trans_points = points + np.array([self.x, self.y])
        return trans_points

    def rotate(self, points:np.ndarray) -> np.ndarray:
        rot_points = np.zeros(shape=(len(points),2))
        rot_points[:,0] = points[:,0] * self.cos - points[:,1] * self.sin
        rot_points[:,1] = points[:,0] * self.sin + points[:,1] * self.cos
        return rot_points
    
    def scale(self, points:np.ndarray) -> np.ndarray:
        sca_points = self.scalar * points
        return sca_points

    def transform(self, points:np.ndarray) -> np.ndarray:
        transform_points = np.zeros(shape=(len(points),2))
        transform_points = self.translate(points)
        transform_points = self.rotate(transform_points)
        return transform_points
    
def fix_rotation(segment:np.ndarray, 
                 points:np.ndarray, 
                 direction='top') -> np.ndarray:
    '''
    Rotates points so that the segment is topmost of bottomost edge
    '''
    new_points = points.copy()

    if direction == 'top':
        max_y = points[:,1].max()
        if max_y > segment[:,1].max():
            # Rotate 180 degrees
            t = Transform(sin=0, cos=-1)
            new_points = t.rotate(new_points)

    elif direction == 'bottom':
        min_y = points[:,1].min()
        if min_y < segment[:,1].min():
            # Rotate 180 degrees
            t = Transform(sin=0, cos=-1)
            new_points = t.rotate(new_points)

    return new_points



def recompute_distance_MST(points, mst):
    n_edges = len(mst)
    for i in range(n_edges):
        e1 = mst[i,0]
        e2 = mst[i,1]
        distance = np.linalg.norm(points[e1,:] - points[e2,:])
        mst[i,2] = distance
    return mst

def compute_mst_mlpack(points, compute_time = False, recompute_distance = False):
    import time
    start = time.time()
    d = mlpack.emst(input_ = points)
    end = time.time()
    total_time = end- start
    d = d["output"]
    mst = np.array([[int(d[i,0]),int(d[i,1]),d[i,2]] for i in range(len(d))],dtype='O')
    if(recompute_distance):
        mst = recompute_distance_MST(points, mst)
    if (compute_time):
        output = (mst,total_time)
    else:
        output = mst
    return output

def compute_mst_ann(points, index, compute_time= False, recompute_distance = False):
    import time 
    start = time.time()
    mst = index.get_mst(points)
    end = time.time()
    total_time = end - start
    mst = mst.tocoo()
    rows = mst.row
    cols = mst.col
    data = mst.data
    mst = np.array([[int(rows[i]),int(cols[i]),data[i]] for i in range(len(rows))],dtype='O')

    if(recompute_distance):
        mst = recompute_distance_MST(points, mst)
    
    if (compute_time):
        output = (mst,total_time)
    else:
        output = mst
    return output

def compute_mst_nx(X, metric='cosine'):
    from sklearn.metrics.pairwise import pairwise_distances
    import networkx as nx

    n = len(X)
    D = pairwise_distances(X, X, metric=metric)

    G = nx.Graph()
    for i in range(n):
        for j in range(1,n):
            if i!=j:
                G.add_edge(i, j, weight=D[i,j])

    T = nx.minimum_spanning_tree(G)

    edges = [[e[0], e[1], e[2]['weight']] for e in T.edges(data=True)]
    edges = np.array(edges, dtype = object)

    return edges


def build_vamana_index(points: np.array,
                       index_directory:str = 'tmp/',
                       vector_dtype = np.float32,
                       distance_metric:str = 'l2',
                       graph_degree:int = 64,
                       complexity:int =100,
                       num_threads:int=0,
                       alpha:float = 1.2,
                       index_prefix:str = 'ann'
                       ):
    '''
    Generate a DiskANN index for a numpy array.

        
    ## Distance Metric and Vector Datatype Restrictions

    | Metric \ Datatype | np.float32 | np.uint8 | np.int8 |
    |-------------------|------------|----------|---------|
    | L2                |      ✅     |     ✅    |    ✅    |
    | MIPS              |      ✅     |     ❌    |    ❌    |
    | Cosine            |      ✅     |     ✅    |    ✅    |

    ### Parameters

    - **points*:  a numpy.ndarray of a supported dtype in 2 dimensions. 
    - **index_directory**: directory that will be used to store the index and temporarily store intermediate files
    - **vector_dtype**: The "points" datatype. If not passed,  we use the `data.dtype` if np array.
    - **distance_metric**: A `str`, strictly one of {"l2", "mips", "cosine"}. `l2` and `cosine` are supported for all 3
      vector dtypes, but `mips` is only available for single precision floats. 
    - **graph_degree**: The degree of the graph index, typically between 60 and 150. A larger maximum degree will
      result in larger indices and longer indexing times, but better search quality.   
    - **complexity**: The size of the candidate nearest neighbor list to use when building the index. Values between 75
      and 200 are typical. Larger values will take more time to build but result in indices that provide higher recall
      for the same search complexity. Use a value that is at least as large as `graph_degree` unless you are prepared
      to compromise on quality
    - **num_threads**: Number of threads to use when creating this index. `0` is used to indicate all available
      logical processors should be used.
    - **alpha**: The alpha parameter (>=1) is used to control the nature and number of points that are added to the
      graph. A higher alpha value (e.g., 1.4) will result in fewer hops (and IOs) to convergence, but probably more
      distance comparisons compared to a lower alpha value.
    - **index_prefix**: The prefix of the index files. Defaults to "ann".

    More information can be found on the library repository https://github.com/microsoft/DiskANN/tree/main/python
    '''
    dir_location = Path(index_directory)
    if(not dir_location.is_dir()):
        os.makedirs(index_directory)
    #remove files is they already exist
    silentremove(os.path.join(index_directory,index_prefix))
    silentremove(os.path.join(index_directory,index_prefix+'.data'))
    silentremove(os.path.join(index_directory,index_prefix+'_metadata.bin'))
    silentremove(os.path.join(index_directory,index_prefix+'_vectors.bin'))

    dap.build_memory_index(
                                data=points,
                                vector_dtype=vector_dtype,
                                distance_metric=distance_metric,
                                index_directory=index_directory,
                                graph_degree=graph_degree,
                                complexity=complexity,
                                num_threads=num_threads,
                                alpha = alpha,
                                index_prefix = index_prefix
                                )

    os.remove(os.path.join(index_directory,index_prefix+'.data'))
    os.remove(os.path.join(index_directory,index_prefix+'_metadata.bin'))
    os.remove(os.path.join(index_directory,index_prefix+'_vectors.bin'))




def build_mst_ann(points: np.array,
                index_directory:str = 'tmp/',
                vector_dtype = np.float32,
                distance_metric:str = 'l2',
                graph_degree:int = 64,
                complexity:int =100,
                num_threads:int=0,
                alpha:float = 1.2,
                index_prefix:str = 'ann',
                tmp_folder = True):
    '''
    Generate a aproximate MST by first computing a  DiskANN's 
    nearest neighbor memory index and then compute the MST over this index

    ### Parameters

    - **tmp_foldeer**:  If True, deletes the directory where the index was built.
    For a description of other parameters, please refer to the documentation of build_vamana_index function

    '''
    build_vamana_index(points = points,
                       index_directory = index_directory,
                       vector_dtype = vector_dtype,
                       distance_metric = distance_metric,
                       graph_degree = graph_degree,
                       complexity = complexity,
                       num_threads = num_threads,
                       alpha = alpha,
                       index_prefix = index_prefix)
    index = Index(os.path.join(index_directory,index_prefix))
    ANN_mst = compute_mst_ann(points,index)
    if(tmp_folder):
        shutil.rmtree(index_directory)
    return ANN_mst