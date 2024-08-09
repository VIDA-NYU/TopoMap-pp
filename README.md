# TopoMap++

Python version of the dimensionality reduction technique called `TopoMap++`. The technique improves upon `TopoMap`, which was originally implemented in C++ [here](https://github.com/harishd10/TopoMap).

`TopoMap` is outlined in the paper:

> Harish Doraiswamy, Julien Tierny, Paulo J. S. Silva, Luis Gustavo Nonato, and Claudio Silva. [TopoMap: A 0-dimensional Homology Preserving Projection of High-Dimensional Data](https://arxiv.org/abs/2009.01512), IEEE Transactions on Visualization and Computer Graphics (IEEE SciVis '20), 2020.

And `TopoMap++` is outlined in:

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

### TopoTree

To run `TopoTree`, you also need to pass your data points as a numpy array. Additionally, `TopoTree` receives the (optional) parameter `min_box_size`, which is the minimum number of points in a component for it to be represented in the tree. In the following example, we set `min_box_size` to be 5% of the data points:

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
