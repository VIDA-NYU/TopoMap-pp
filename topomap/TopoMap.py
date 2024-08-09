import numpy as np 
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from .utils import closest_edge_point_simplices, find_angle, Transform, fix_rotation, build_mst_ann, compute_mst_mlpack, compute_mst_nx

from .UnionFindComponents import UnionFind

class TopoMap(BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin):
    """Generate TopoMap projection of the input data points.

    Parameters
    ----------

    metric : {'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',' braycurtis', 'canberra', \
                'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', \
                'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',\
                'sqeuclidean', 'yule'} or a callable, default='euclidean'
        Distance metric to compute the MST. Check out sklearn's pairwise_distances to see other options.

    approach: {'mlpack', 'nx', 'ANN'}, default='mlpack'
        Approach for computing the MST. If 'mlpack', mlpack's function emst is used. In this case, the only 
        metric available is euclidean. If 'nx', it uses sklearn's pairwise_distance function followed by 
        networkx's minimum_spanning_tree function. If 'ANN', it first compute a  DiskANN's 
        nearest neighbor memory index and then compute the MST over this index. In  this case, only "euclidean", 
        "l2", "mips" and "cosine" are available

    load_mst: Bool, default=False
        Option to load precomputed mst. If set to True, you need to provide the mst_path.

    mst_path: string, default=''
        Path to the precomputed mst. Only used if load_mst is True.

    The following parameters are only used if approach == 'ANN'

    - **index_directory**: directory that will be used to store the index and temporarily store intermediate files
    - **vector_dtype**: The "points" datatype. If not passed,  we use the `data.dtype` if np array. 
    | Metric \ Datatype | np.float32 | np.uint8 | np.int8 |
    |-------------------|------------|----------|---------|
    | L2 | Euclidean    |      ✅     |     ✅    |    ✅    |
    | MIPS              |      ✅     |     ❌    |    ❌    |
    | Cosine            |      ✅     |     ✅    |    ✅    |
    
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
    - **tmp_foldeer**:  If True, deletes the directory where the index was built.
        
    """

    def __init__(self, 
                 metric='euclidean', 
                 approach = 'mlpack', 
                 load_mst = False, 
                 mst_path = '',
                 index_directory:str = 'tmp/',
                vector_dtype = np.float32,
                graph_degree:int = 64,
                complexity:int =100,
                num_threads:int=0,
                alpha:float = 1.2,
                index_prefix:str = 'ann'
                 ) -> None:
        self.metric = metric
        self.approach = approach
        self.load_mst  = load_mst
        self.mst_path = mst_path
        self.index_directory = index_directory
        self.vector_dtype = vector_dtype
        self.graph_degree = graph_degree
        self.complexity = complexity
        self.num_threads = num_threads
        self.alpha = alpha
        self.index_prefix = index_prefix

        if metric!='euclidean' and approach=='mlpack':
            self.approach = 'nx'

        # Parameters to set if they were precomputed
        self.mst = None
        self.sorted_edges = None

    def _compute_mst(self, X):
        if self.load_mst:
            mst = np.load(self.mst_path,allow_pickle= True)
        elif self.approach == 'mlpack':
            mst = compute_mst_mlpack(X)
        elif self.approach == 'nx':
            mst = compute_mst_nx(X, metric=self.metric)
        elif self.approach == 'ANN':
            metric = self.metric
            if(metric == 'euclidean'):
                metric = 'l2'
            elif(metric not in ['l2','mips', 'cosine']):
                print("only euclidean, mips, cosine and l2 metrics are supported for ANN approach")
            #Build VamanaIndex and mst
            mst = build_mst_ann(points = X,
                                index_directory = self.index_directory,
                                vector_dtype = self.vector_dtype,
                                distance_metric = metric,
                                graph_degree = self.graph_degree,
                                complexity = self.complexity,
                                num_threads = self.num_threads,
                                alpha = self.alpha,
                                index_prefix = self.index_prefix 
                                )
        else:
            print("Approach isn't supported, please choose between ANN, mlpack and nx")
        return mst
      
    def _compute_ordered_edges(self, mst):
        sorted_edges = mst[mst[:, 2].argsort()]
        return sorted_edges
    
        
    def _get_mst(self):
        return self.mst
        
    def _get_sorted_edges(self):
        return self.sorted_edges

    def _rotate_component(self, 
                         component_ids:list,
                         ref_point:np.ndarray, 
                         component_simplices:list,
                         direction='top') -> np.ndarray:

        component_points = self.projections[component_ids,:]

        if len(component_simplices)==1:
            closest_edge = self.projections[component_simplices[0]]
            edge_i = component_simplices[0]
        else:
            closest_edge, edge_i = closest_edge_point_simplices(component_simplices,
                                                        self.projections,
                                                        ref_point)
            
        t = Transform()
        t.cos, t.sin = find_angle(closest_edge)
        component_points = t.rotate(component_points)

        id_0 = component_ids.index(edge_i[0])
        id_1 = component_ids.index(edge_i[1])
        closest_edge = np.array([component_points[id_0,:],
                                    component_points[id_1,:]])
        component_points = fix_rotation(closest_edge, 
                                        component_points, 
                                        direction=direction)

        return component_points, [id_0,id_1]
    
    def _translate_component(self, 
                            component_points:list, 
                            edge_i:np.ndarray,
                            to_point:list) -> np.ndarray:
        
        if component_points[edge_i[0], 0] <= component_points[edge_i[1], 0]:
            t = Transform(x = to_point[0]-component_points[edge_i[0], 0], 
                          y = to_point[1]-component_points[edge_i[0], 1])
            component_points = t.translate(component_points)

        else:
            t = Transform(x = to_point[0]-component_points[edge_i[1], 0], 
                          y = to_point[1]-component_points[edge_i[1], 1])
            component_points = t.translate(component_points)

        return component_points

    def _project_points(self):
        
        for i in range(len(self.sorted_edges)):
            # Get points from the edge
            i_a, i_b = self.sorted_edges[i][0], self.sorted_edges[i][1]
            p_a, p_b = self.projections[i_a,:], self.projections[i_b,:]

            # Distance between points
            d = self.sorted_edges[i][2]
            
            # Get components the points belong to
            root_a, root_b = self.uf.find(i_a), self.uf.find(i_b)
            c_a = self.uf.component[root_a].points_ids.copy()
            c_b = self.uf.component[root_b].points_ids.copy()

            proj_c_a, proj_c_b = self.projections[list(c_a)], self.projections[list(c_b)]

            simplices_a = self.uf.component[root_a].hull_simplices
            simplices_b = self.uf.component[root_b].hull_simplices

            # Rotate the first to be the topmost
            proj_c_a, edge_t = self._rotate_component(c_a, p_a, simplices_a, direction='top')
            # Rotate the second to be the bottomost
            proj_c_b, edge_b = self._rotate_component(c_b, p_b, simplices_b, direction='bottom')

            # Translate components
            proj_c_a = self._translate_component(proj_c_a, edge_t, to_point=[0,0])
            proj_c_b = self._translate_component(proj_c_b, edge_b, to_point=[0,d])

            self.projections[list(c_a), :] = proj_c_a
            self.projections[list(c_b), :] = proj_c_b

            # Merge components 
            self.uf.union(i_a, i_b, self.projections)

        return self.projections
    
    def fit_transform(self, X:np.ndarray):
        """Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in two-dimensional space.
        """

        self.projections = np.zeros(shape=(len(X), 2), dtype=np.float32)
        self.uf = UnionFind(len(X))

        if self.mst is None:
            self.mst = self._compute_mst(X)

        if self.sorted_edges is None:
            self.sorted_edges = self._compute_ordered_edges(self.mst)

        self.projections = self._project_points()

        return self.projections
