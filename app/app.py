import sys
from flask import Flask, render_template, request, jsonify,Response
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import pandas as pd 
import numpy as np
import json
import pickle
import os 

sys.path.append('../')
sys.path.append('../..')
from topomap  import TopoTree
from topomap import HierarchicalTopoMap
from topomap import TopoMap
from topomap.utils import  get_hull,build_mst_ann
from pathlib import Path



def plot_projections_discrete_feature(projections,
                                    df_data,
                                    column_color,
                                    column_values=[],
                                    colors=[],
                                    hiertopomap_points=None,
                                    components_to_scale = [],
                                    legend_title='',
                                    low_opacity=False,
                                    show_hulls=True
                                    ):
    if len(column_values) == 0:
        column_values = list(df_data[column_color].unique())
    if len(colors)==0:
        colors = px.colors.qualitative.T10
    if legend_title=='':
        legend_title=column_color

    fig = go.Figure()

    if show_hulls:
        for c in components_to_scale:
            comp_ids = hiertopomap_points[c]

            hull = get_hull(projections[comp_ids,:])
            points_ids = [comp_ids[i] for i in hull.vertices]
            points = list(projections[points_ids,:])
            points.append(points[0])
            xs, ys = zip(*points)

            fig.add_trace(go.Scatter(x=xs, y=ys,
                            fill='toself', 
                            fillcolor = '#CCCCCC',
                            line_color='#808080',
                            opacity=0.5,
                            line_width=1,
                            text=f'Component {c}',
                            name='Components', legendgroup='Components',
                            showlegend=False,
                            marker=dict(size=1)
                            )
                        )

    if low_opacity:
        opacity_points = 0.5
    else:
        opacity_points = 1

    for i,c in enumerate(column_values):
        points_id = df_data[df_data[column_color]==c].index.to_list()
        fig.add_trace(go.Scatter(x=projections[points_id,0], 
                                y=projections[points_id,1],
                                customdata=points_id,
                                mode='markers',
                                opacity=opacity_points,
                                marker=dict(
                                    color=colors[i],
                                    size=2,
                                ),
                                name=str(c),
                                showlegend=True
                                )
            )
        
    fig.update_layout(margin = dict(t=25, l=25, r=25, b=25),
                        # height=500,
                        # width=550,
                        legend= {'itemsizing': 'constant',
                        'title': legend_title},
                        plot_bgcolor = "white",
                        xaxis=dict(showticklabels=False,
                        showline=True, linewidth=1, linecolor='black', mirror=True), 
                        yaxis=dict(showticklabels=False,
                        showline=True, linewidth=1, linecolor='black', mirror=True),
                    )
    
    return fig



app = Flask(__name__)
cur_plot = 0
data_folder = 'data_app/numpy_datasets_app/'
#Features that can be used to color the projection. Currently we only accept categorial features 
feature_folder = 'data_app/features_app/'
hierarquical_folder = 'data_app/hierarquical_saved_app/'
proj_folder = 'data_app/proj_saved_app/'
mst_folder = 'data_app/msts_app/'
ordered_edges_folder = 'data_app/ordered_edges_app/'
df_comp_folder = 'data_app/saved_dfcomp_app/'

Path(feature_folder).mkdir(parents=True, exist_ok=True)
Path(hierarquical_folder).mkdir(parents=True, exist_ok=True)
Path(proj_folder).mkdir(parents=True, exist_ok=True)
Path(mst_folder).mkdir(parents=True, exist_ok=True)
Path(ordered_edges_folder).mkdir(parents=True, exist_ok=True)
Path(df_comp_folder).mkdir(parents=True, exist_ok=True)



datasets = {
    "LLM":"llm_embs.npy",
    "Iris":"Iris.npy"
    
}

#Path to features of each possible dataset
features = {
    "Iris":"target_Iris.csv",
    "LLM":"llm_df_questions.csv"
}

#List of possible categorical columns to use to color the projection.
available_columns = { 
    "Iris":["class"],
    "LLM":["model_choice", 'Level', 'correct','correct_answer']
}




def plot_topomap(projections,
                 df_data,
                column_color,
                column_values=[],
                colors=[],
                legend_title='',
                low_opacity=False,
                 ):
    
    if low_opacity:
        opacity_points = 0.5
    else:
        opacity_points = 1
    
    if len(column_values) == 0:
        column_values = list(df_data[column_color].unique())
    if len(colors)==0:
        colors = px.colors.qualitative.T10
    if legend_title=='':
        legend_title=column_color
    fig = go.Figure()

    for i,c in enumerate(column_values):
        points_id = df_data[df_data[column_color]==c].index.to_list()
        fig.add_trace(go.Scatter(x=projections[points_id,0], 
                                y=projections[points_id,1],
                                customdata=points_id,
                                mode='markers',
                                opacity=opacity_points,
                                marker=dict(
                                    color=colors[i],
                                    size=2,
                                ),
                                name=str(c),
                                showlegend=True
                                )
            )
    
   

    

    fig.update_layout(margin = dict(t=25, l=25, r=25, b=25),
                        legend= {'itemsizing': 'constant',
                        'title': legend_title},
                        plot_bgcolor = "white",
                        xaxis=dict(showticklabels=False,
                        showline=True, linewidth=1, linecolor='black', mirror=True), 
                        yaxis=dict(showticklabels=False,
                        showline=True, linewidth=1, linecolor='black', mirror=True),
                        title = "Topomap projection"
                    )

    return fig

def get_hierarquical(dataset,eta):
    hieraquical_path = f'{hierarquical_folder}Hierarquical_{dataset}_{eta}.pickle'
    with open(hieraquical_path, 'rb') as f:
        return pickle.load(f)
    

def check_hierarquical(dataset,eta):
    hieraquical_path = f'{hierarquical_folder}Hierarquical_{dataset}_{eta}.pickle'
    return os.path.isfile(hieraquical_path)

def check_mst(dataset):
    mst_path = f'{mst_folder}mst_{dataset}.npy'
    return os.path.isfile(mst_path)

def get_mst(dataset,X):
    mst_path = f'{mst_folder}mst_{dataset}.npy'
    if(check_mst(dataset)):     
        with open(mst_path, 'rb') as f:
            return np.load(f,allow_pickle=True)
    else: 
        mst = build_mst_ann(X)
        with open(mst_path, 'wb') as f:
            np.save(f,mst,allow_pickle=True)
        return mst

    

def check_ordered_edges(dataset):
    oe_path = f'{ordered_edges_folder}ordered_edges_{dataset}.npy'

    return os.path.isfile(oe_path)

def get_ordered_edges(dataset,X):
    oe_path = f'{ordered_edges_folder}ordered_edges_{dataset}.npy'
    if(check_ordered_edges(dataset)):
        with open(oe_path, 'rb') as f:
            return np.load(f, allow_pickle=True)
    else:
        mst = get_mst(dataset,X)
        sorted_edges = mst[mst[:, 2].argsort()]
        with open(oe_path, 'wb') as f:
            np.save(f,sorted_edges,allow_pickle=True)
        return sorted_edges

    



def find_root(df):
    id = 0 
    parent = int(df.iloc[0]["parent"])
    while(True):
        id = int(parent)
        try:
            parent = int(df.iloc[parent]["parent"])
        except:
            break
    return id

def populate_tree(df,node):
    node = int(node)
    base_dict ={
        "persistence":df.iloc[node]["died_at"] if not pd.isna(df.iloc[node]["died_at"]) else 'nan',
        "name": str(df.iloc[node]["id"]),
        "value": df.iloc[node]["size"]
    }
    
    childrens = df[df["parent"] ==node]["id"].to_list()
    if(len(childrens)==0):
        return base_dict
    else:
        aux_list = []
        for child in childrens:
            aux_list.append(populate_tree(df,child))
        base_dict["children"] = aux_list
        return base_dict
    

def transform_df_d3(df):
    root = find_root(df)
    return populate_tree(df,root)



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return super(NpEncoder, self).default(obj)



    
    


def check_dfcomp(dataset,eta):
    df_comp_path = f'{df_comp_folder}{dataset}_{eta}.parquet'
    return os.path.isfile(df_comp_path)

def get_df_comp(dataset,eta,X):
    if(check_dfcomp(dataset,eta)):
        df_comp = pd.read_parquet(f'{df_comp_folder}{dataset}_{eta}.parquet')
    else:
        topotree = TopoTree(min_box_size=eta)
        topotree.mst = get_mst(dataset, X)
        topotree.sorted_edges = get_ordered_edges(dataset,X)
        comp_info = topotree.fit(X)
        df_comp = pd.DataFrame.from_dict(comp_info)
        df_comp.to_parquet(f'{df_comp_folder}{dataset}_{eta}.parquet', index = False)
        
    return df_comp


def check_all_childs(node, node_list):
    if('children' not in node.keys()):
        return int(node['name']) in node_list
    if int(node['name']) not in node_list:
        return False
    else: 
        return check_all_childs(node['children'][0],node_list) and check_all_childs(node['children'][1], node_list)
    
def find_higher_node(tree, node_list):
    if( int(tree['name']) in node_list):
        
        return [tree]
    
    if('children' not in tree.keys()):
       return []

         
    
    return find_higher_node(tree['children'][0], node_list) + find_higher_node(tree['children'][1], node_list) 


def get_custom_component(df_comp,tree,
                          node_list,
                         points):
    if("children" not in tree.keys()):
        node_id = int(tree["name"])
        if(node_id in node_list ):
            points  = points.union(set(df_comp.loc[node_id]["points"]))
        else:
            points = points.difference(set(df_comp.loc[node_id]["points"]))
        return points
    
    node_id = int(tree["name"])
    if(node_id in node_list ):
        points  = points.union(set(df_comp.loc[node_id]["points"]))
    else:
        points = points.difference(set(df_comp.loc[node_id]["points"]))

    points =  get_custom_component(df_comp, tree["children"][0], node_list, points)
    points = get_custom_component(df_comp, tree["children"][1], node_list, points)
    return points



@app.route('/')
def index():

    return render_template('index.html')

@app.route('/plotly-plot', methods=['POST'])
def plotly_plot():
    data = request.get_json()
    
    selected_names = data.get('global_selecteds', [])
    selected_names = [int(s) for s in selected_names]
    dataset_name = data.get('dataset_name', [])
    column_to_color = data.get('column_to_color','')
    X_path = data_folder + datasets[dataset_name]
    y_path = feature_folder + features[dataset_name]
    eta = data.get('eta', '')
    if(eta ==''):
        eta = 0
    else:
        eta = int(eta)
    with open(X_path, 'rb') as f:
        X = np.load(f)
    
    y = pd.read_csv(y_path)
    X=X.astype(np.float32)
    
    df_comp = get_df_comp(dataset_name,eta,X)
    
    components_to_scale = selected_names
    
    if(column_to_color in ['correct_answer', 'model_choice']):
        column_values=['A','B','C','D']
    else:
        column_values = []

    if(len(components_to_scale)!=0):
        sorted_components = "_".join(map(str, sorted(components_to_scale)))
        proj_path = f'{proj_folder}Hierarchical_proj_{dataset_name}_eta_{eta}_components_{sorted_components}.npy'
        hier_path = f'{hierarquical_folder}Hierarchical_{dataset_name}_eta_{eta}_components_{sorted_components}.parquet'
        if(os.path.isfile(proj_path)):
            with open(proj_path,'rb') as f:
                proj_hier = np.load(f,allow_pickle=True)
            
            hiertopomap_points = pd.read_parquet(hier_path)["points"].to_list()
            
        else:
            hiertopomap = HierarchicalTopoMap()
            hiertopomap.min_points_component = eta
            hiertopomap.components_to_scale = components_to_scale
            hiertopomap.df_comp = df_comp 
            hiertopomap.approach = "ANN"
            hiertopomap.mst = get_mst(dataset_name, X)
            hiertopomap.sorted_edges  =get_ordered_edges(dataset_name,X)
            proj_hier = hiertopomap.fit_transform(X)
            with open(proj_path, 'wb') as f: 
                np.save(f,proj_hier,allow_pickle = True)
            
            df = pd.DataFrame(hiertopomap.components_info, columns=['points'])

            df.to_parquet(hier_path, index = False)
            hiertopomap_points = df["points"].to_list()
            

        
        fig = plot_projections_discrete_feature(proj_hier,
                                            y,
                                            column_color = column_to_color,
                                            column_values=column_values,
                                            legend_title=f"{dataset_name} {column_to_color}",
                                            low_opacity=True,
                                            hiertopomap_points=hiertopomap_points,
                                            components_to_scale= components_to_scale)
    
        
        fig.update_layout(autosize= True, title = "Topomap++ projection")
    
    else:
        proj_path = f'{proj_folder}Topomap_proj_{dataset_name}.npy'
        if(os.path.isfile(proj_path)):
            with open(proj_path,'rb') as f:
                proj_topomap = np.load(f, allow_pickle=True)
        else:
            topomap= TopoMap(approach = 'ANN')
            topomap.mst = get_mst(dataset_name, X)
            proj_topomap = topomap.fit_transform(X)
            with open(proj_path,'wb') as f:
                np.save(f, proj_topomap,allow_pickle=True)
        fig = plot_topomap(proj_topomap,y,
                                    column_color = column_to_color,
                                    column_values=column_values,
                                    legend_title=f"{dataset_name} {column_to_color}",
                                    low_opacity=True)
        fig.update_layout( autosize= True)
    global cur_plot 

    cur_plot= fig
    graph_json = fig.to_json()

    return jsonify(graph_json)

@app.route('/get_leaves',methods = ['POST'])
def get_leaves():
    data = request.get_json()
    dataset = data.get('dataset_name', '')
    eta = data.get('eta', '')
    if(eta ==''):
        eta = 0
    else:
        eta = int(eta)
    
    X_path = data_folder + datasets[dataset]
   
    with open(X_path, 'rb') as f:
        X = np.load(f)

    X=X.astype(np.float32)
    
    hiertopomap = HierarchicalTopoMap()
    df_comp = get_df_comp(dataset,eta,X)
   
    components = hiertopomap._get_component_to_scale(X = [],df_comp= df_comp,selection_method='min_size')
    
    return jsonify({'ok': True, 
        'msg':'Success',"components":components} )


@app.route('/get-dataset', methods=['POST'])
def get_dataset():
    data = request.get_json()
    eta = int(data.get('eta', ''))
    dataset_name = data.get('dataset_name', '')
    X_path = data_folder + datasets[dataset_name]

    with open(X_path, 'rb') as f:
        X = np.load(f)
    

    X=X.astype(np.float32)
    
    df_comp = get_df_comp(dataset_name,eta,X)
    
  
    json_df_comp = transform_df_d3(df_comp)
    

    return Response(json.dumps(json_df_comp, cls=NpEncoder))



@app.route('/columns', methods=['GET'])
def get_columns():
    dataset_name = request.args.get('dataset')
    if dataset_name in datasets:
        columns = available_columns[dataset_name]
        return jsonify({'columns': columns})
    else:
        return jsonify({'error': 'Dataset not found'}), 404

@app.route('/datasets', methods=['GET'])
def get_datasets():
    return jsonify({'datasets': list(datasets.keys())})

if __name__ == '__main__':
    app.run(debug=True)
