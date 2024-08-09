# TopoMap++

GitHub repository for our paper, *TopoMap++: A faster and more space efficient technique to compute projections with topological guarantees*. TopoMap++ is a layout improvement scheme to highlight important structures in the TopoMap layout. This new approach maintains local topological guarantees and introduces an interactive, TreeMap-based visualization for easier analysis of high-dimensional datasets. Additionally, we propose an efficient approximation scheme inspired by ANNS algorithms to compute the Rips filtration, drastically reducing computational costs while preserving data topology.

TopoMap++ improves upon TopoMap, which was originally implemented in C++ [here](https://github.com/harishd10/TopoMap) and outlined in the paper:

> Harish Doraiswamy, Julien Tierny, Paulo J. S. Silva, Luis Gustavo Nonato, and Claudio Silva. [TopoMap: A 0-dimensional Homology Preserving Projection of High-Dimensional Data](https://arxiv.org/abs/2009.01512), IEEE Transactions on Visualization and Computer Graphics (IEEE SciVis '20), 2020.

And TopoMap++ is outlined in:

> Vitoria Guardieiro, Felipe Inagaki de Oliveira, Harish Doraiswamy, Luis Gustavo Nonato, and Claudio Silva. TopoMap++: A faster and more space efficient technique to compute
projections with topological guarantees, IEEE Transactions on Visualization and Computer Graphics (IEEE VIS '24), 2024.

## Usage

This version was implemented with Python 3.11.7 and all packages are listed in `requirements.txt`.

### TopoMap

To run `TopoMap`, you just need to pass your data points as a numpy array (`X` in the following example):

```
from TopoMap import TopoMap

topomap = TopoMap()
proj = topomap.fit_transform(X)
```

The output `proj` is also a numpy array, with the same number of rows as `X` and two dimensions.

To use an approximate but much faster version, set approach="ANN" when creating the TopoMap, like this: `TopoMap(approach="ANN")`. Note that this requires the data to be in `np.float32`, `np.uint8`, or `np.int8` format.

### TopoTree


To run `TopoTree`, you also need to pass your data points as a numpy array. Additionally, `TopoTree` receives the (optional) parameter `min_box_size`, which is the minimum number of points in a component for it to be represented in the tree. In the following example, we set `min_box_size` to 5% of the data points:

```
from TopoTree import TopoTree

topotree = TopoTree(min_box_size=0.05*X.shape[0])
comp_info = topotree.fit(X) 
```

The output `comp_info` is a list in which each element is a dictionary corresponding to a component and containing information such as its id, size, and list of data points.

To visualize the components as a tree, we provide the `plot_hierarchical_treemap` function in `visualizations.py`. To do so, you will need to pass the components' information as a pandas DataFrame:

```
from visualizations import plot_hierarchical_treemap

df_comp = pd.DataFrame.from_dict(comp_info)
fig = plot_hierarchical_treemap(df_comp_blobs)
fig.show()
```

### Hierarchical TopoMap

To run `HierarchicalTopoMap`, you also need to pass your data points as a numpy array. Additionally, you need to indicate which components to scale (by providing a list of component ids) or how the component selection should be made:

```
from HierarchicalTopoMap import HierarchicalTopoMap

hier_topomap = HierarchicalTopoMap(components_to_scale=components_to_scale)
proj = hier_topomap.fit_transform(X)
```

## Examples and Case Studies

In the "examples" folder, we provide example notebooks that illustrate the use of Topomap, TopoTree, and Hierarchical Topomap. These notebooks also demonstrate how to compute the approximate minimum spanning tree and compare it with the original.

Additionally, we offer notebooks that reproduce the case studies section of our paper.

To reproduce the outputs in the notebooks, please place our [data](https://drive.google.com/file/d/1unPHq1-wc_nODQP2igb-28peXbyFn-NN/view?usp=sharing), in the 'data' folder at the root of the project. The StreetAware data can be found [here](https://drive.google.com/drive/folders/1nkmWsjCDIDws4zL7WMRcLiOn2qqroRsE?usp=sharing)

## App

We also provide a simple interface for iteratively exploring the connection between a Hierarchical Topomap and a TopoTree. To use it, please install Flask and Fastparquet.

We have an [app_data](https://drive.google.com/file/d/1RdLbcOBsedBO6LQ9u8IX_-SNC72VOFlS/view?usp=sharing) folder containing the necessary structure and data examples. To use your own data, add your dataset as a NumPy array in the "numpy_datasets_app" subfolder and your dataframe of features as a CSV in the "features_app" subfolder. Then, update the "datasets," "features," and "available_columns" dictionaries in app.py accordingly.
