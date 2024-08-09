import numpy as np
import pandas as pd
import pickle
import os
import time

np.random.seed(2024)

import plotly.express as px

import sys
sys.path.append('../..')

from topomap import TopoMap, TopoTree, HierarchicalTopoMap

# Load data
df_questions_path = 'ARC_df_questions.csv'
embs_path = 'ARC_embeddings.pkl'

if os.path.isfile(df_questions_path):
    df_questions = pd.read_csv(df_questions_path)
    with open(embs_path, 'rb') as f:
        embs = pickle.load(f)

else:
    levels = ['Easy', 'Challenge']
    sets = ['train', 'test']

    numbers = ['1','2','3','4']

    df_questions = pd.DataFrame()
    embs = []

    for level in levels:
        for set in sets:
            df_set_level = pd.read_csv(f'../../data/LLM/ARC_{level}_{set}_questions_data.csv')
            df_set_level['Level'] = level
            df_set_level = df_set_level[~df_set_level['correct_answer'].isin(numbers)]
            non_number_ids = df_set_level.index.to_list()
            df_questions = pd.concat([df_questions, df_set_level], ignore_index=True)

            emb_all_layers = pickle.load(open(f'../../data/LLM/ARC_{level}_{set}_layers_emb.pkl', 'rb'))
            emb_last_layer = emb_all_layers[-1]
            emb_last_layer_cleaned = []
            for i in non_number_ids:
                emb_last_layer_cleaned.append(emb_last_layer[i])
            embs.extend(emb_last_layer_cleaned)

    embs = np.array(embs)

    mask = df_questions.map(type) != bool
    d = {True: 'True', False: 'False'}
    df_questions = df_questions.drop('Unnamed: 0', axis=1)
    df_questions = df_questions.where(mask, df_questions.replace(d))

    df_questions.to_csv(df_questions_path, index=False)
    with open(embs_path, 'wb') as f:
        pickle.dump(embs, f)


## TopoMap
topomap_path = "topomap_llm.pkl"

if os.path.isfile(topomap_path):
    with open(topomap_path, 'rb') as f:
        topomap_llm = pickle.load(f)
    proj_topomap_llm = topomap_llm.projections
else:
    start_time = time.time()

    topomap_llm = TopoMap()
    proj_topomap_llm = topomap_llm.fit_transform(embs)

    topomap_time = time.time()-start_time
    print(f'Time for running TopoMap: {topomap_time:.3f}s')

    with open(topomap_path,'wb') as f:
        pickle.dump(topomap_llm, f)


## HierarchicalTopoMap
start_time = time.time()

hiertopomap_llm = HierarchicalTopoMap(k_components=5)
hiertopomap_llm.min_points_component = 50
hiertopomap_llm.mst = topomap_llm.mst
hiertopomap_llm.sorted_edges = topomap_llm.sorted_edges
proj_hier_llm = hiertopomap_llm.fit_transform(embs)

hier_time = time.time()-start_time
print(f'Time for running HierarchicalTreeMap: {hier_time:.3f}s')


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from wordcloud import WordCloud
from io import BytesIO
import base64

cs = px.colors.qualitative.D3
color_map_feat = {'model_choice': {'A':cs[0], 'B':cs[1], 'C':cs[2], 'D':cs[3], 'E': cs[8]},
                  'correct_answer': {'A':cs[0], 'B':cs[1], 'C':cs[2], 'D':cs[3], 'E': cs[8]},
                  'correct': {True:px.colors.qualitative.Plotly[0], 
                              False:px.colors.qualitative.Plotly[1]},
                  'Level': {'Easy': px.colors.qualitative.T10[3],
                            'Challenge': px.colors.qualitative.T10[1]}
                }

def hover_text(df, i):
    return f"Index: {i}<br>Model choice: {df.iloc[i]['model_choice']}<br>Correct choice: {df.iloc[i]['correct_answer']}<br>Correct: {df.iloc[i]['correct']}<extra></extra>"

def get_color(color, value):
    return color_map_feat[color][value]

def get_colors(data, color):
    return data[color].map(color_map_feat[color])


def get_scatter(df, proj, color, selected_id, legend=True, width=800,
                hover_data=True, markersize=2):   
    fig = go.Figure()
    for cat in list(color_map_feat[color].keys()):
        df_cat = df[df[color] == cat]
        ids = list(df_cat.index)
        if len(ids) == 0:
            continue

        marker = {'color': get_color(color, cat)}
        if not markersize is None:
            marker['size'] = markersize

        fig.add_trace(go.Scatter(x=proj[ids,0], y=proj[ids,1], 
                                 marker=marker,
                                 hovertemplate = "%{text}",
                                 text=[hover_text(df, i) for i in ids],
                                 mode='markers',
                                 customdata=ids,
                                 name=cat,
                                 )
                     )

    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.08,
                            xanchor="right",
                            x=1.0,
                            title=color
                        ),
                    margin=dict(l=0.1,r=0.1,b=0.1,t=0.1),
                    width=1600,
                    height=800,
                    showlegend=legend,
                    dragmode='select'
                    )
    if not selected_id is None:
        fig.update_traces(selectedpoints=[])

        colors = get_colors(df.iloc[selected_id], color)

        marker = {'color': colors}
        if not markersize is None:
            marker['size'] = markersize

        fig.add_scatter(x=proj[selected_id,0], y=proj[selected_id,1],
                        marker=marker,
                        hovertemplate = "%{text}",
                        text=[hover_text(df, i) for i in selected_id],
                        mode='markers',
                        customdata=selected_id,
                        showlegend=False
                        )
        
    if not hover_data:
        fig.update_layout(hovermode=False)

    return fig

def get_cat_bar(df, color, selected_id, showlegend=True, title=None):
    orig_prop = df.groupby(color).count()/df.shape[0]

    if title is None:
        title = 'Proportion of questions per '+color

    if selected_id is None:
        selected_prop = orig_prop
    else:
        selected_prop = df.iloc[selected_id].groupby(color).count()/len(selected_id)

    fig = go.Figure(data=[
        # go.Bar(name='All questions', 
        #        x=list(orig_prop.index), 
        #        y=list(orig_prop['question']),
        #        marker_color=px.colors.qualitative.Plotly[0]
        #     ),
        go.Bar(name='Selection', 
               x=list(selected_prop.index), 
               y=list(selected_prop['question']),
               marker_color=px.colors.qualitative.Plotly[0]
            )
    ])
    fig.update_layout(barmode='group',
                      title=title,
                      yaxis=dict(title='Proportion of questions'),
                      xaxis=dict(title=color),
                      showlegend=False,
                      legend=dict(
                            x=0,
                            y=1.0,
                            bgcolor='rgba(255, 255, 255, 0)',
                            bordercolor='rgba(255, 255, 255, 0)'
                        ),
                      margin=dict(l=70, r=20, t=50, b=20),
                      font_family="Arial",
                      height=400
                     )

    return fig

def get_wordcloud(df, selected_ids=None):
    if selected_ids is None:
        selected_ids = df.index

    text = df.iloc[selected_ids].question.str.cat(sep=' ')
    for choice in list(color_map_feat['correct_answer'].keys()):
        text = text.replace(f'({choice})', ' ')

    wordcloud = WordCloud(width=1080, height=360, background_color="white",
                          random_state=1)
    wordcloud.generate(text)

    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())



from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
            suppress_callback_exceptions=True)

map_col = {'Unnamed: 0': 'Index', 'question': 'Question', 
           'model_choice': 'Model answer', 'correct_answer': 'Correct answer', 'correct': 'Correct',
           'Level': 'Level'}

app.layout = [

    html.H4("Projection Explorer"),
    html.Hr(),

    dbc.Row([
        dbc.Col([
            html.Div(className='options-left', children=[
                html.Div(className='options-left-color', children=[
                        dbc.Select(['correct_answer', 'model_choice', 'correct', 'Level'],
                        value='model_choice', id='control-left-color')
                ]),
            ], style={'display': 'flex', 'flexDirection': 'row'}),

            dcc.Graph(id='scatter-left', figure=get_scatter(df_questions, 
                                                            proj_hier_llm,
                                                            'model_choice',
                                                            None,
                                                            legend=True)),
        ],
        style={'marginBottom': '1em',
            'backgroundColor': '#E6E6E6'}),
    ], style={'marginBottom': '1em',
                'marginTop':    '2em',
                'marginLeft':   '3em',
                'marginRight':  '3em',
                'backgroundColor': '#E6E6E6'}),

    dbc.Accordion([
        dbc.AccordionItem([
                dbc.Row([
                        dbc.Col([dcc.Graph(id='bar-left', figure=get_cat_bar(df_questions, 'correct_answer', None))], 
                                ),
                        dbc.Col([dcc.Graph(id='bar-mid', figure=get_cat_bar(df_questions, 'model_choice', None))], 
                                ),
                        dbc.Col([dcc.Graph(id='bar-right', figure=get_cat_bar(df_questions, 'correct', None))], 
                                width=2,
                                ),
                        dbc.Col([dcc.Graph(id='bar-right-r', figure=get_cat_bar(df_questions, 'Level', None))], 
                                width=2,
                                ),
                        
                ]),
        ], title = "Proportion of categories"),

        dbc.AccordionItem([
                dbc.Row([
                    dash_table.DataTable(data=df_questions.to_dict('records'),
                                        columns=[{'id': c, 'name': map_col[c]} for c in df_questions.columns],
                                        page_size=10,
                                        style_cell={'textAlign': 'center',
                                                    'font-family':'sans-serif',
                                                    },
                                        style_cell_conditional=[
                                                {
                                                    'if': {'column_id': c},
                                                    'textAlign': 'left'
                                                } for c in ['question']
                                            ],
                                        style_data={
                                                'whiteSpace': 'pre-line',
                                                'height': 'auto',
                                        },
                                        style_data_conditional=[
                                                {
                                                    'if': {'row_index': 'odd'},
                                                    'backgroundColor': 'rgb(220, 220, 220)',
                                                }
                                            ],
                                        style_header={'fontWeight': 'bold',
                                                        'textAlign': 'center'},
                                        id='table')

                ]),
            ], title = "Question Table"),

        dbc.AccordionItem([
                dbc.Row([dbc.Col([html.H5("All questions"),
                        html.Img(id="wc_all", src=get_wordcloud(df_questions, None), style={'width': '100%'})], 
                        style={'marginBottom': '1em'}),
                dbc.Col([html.H5("Selected questions"),
                        html.Img(id="wc_selected", src=get_wordcloud(df_questions, None), style={'width': '100%'})], 
                        style={'marginBottom': '1em'}),
                ]),
        ], title = "Wordclouds"),
    ],
    style={'marginBottom': '1em',
                'marginTop':    '2em',
                'marginLeft':   '3em',
                'marginRight':  '3em'
                },
        always_open=True
        ),

]

@callback(
    Output(component_id='scatter-left', component_property='figure'),
    [Input(component_id='control-left-color', component_property='value')]
)
def update_scatter_left(color):
    fig_left = get_scatter(df_questions, proj_hier_llm, color, None, legend=True)
    return fig_left

@callback(
    [Output(component_id='bar-left', component_property='figure'),
     Output(component_id='bar-mid', component_property='figure'),
     Output(component_id='bar-right', component_property='figure'),
     Output(component_id='bar-right-r', component_property='figure'),
    ],
    [Input("scatter-left", "selectedData")],
    allow_duplicate=True
)
def update_bar_plots(selection1):
    if selection1 is None:
        selected_id = None
    else:
        selected_id = []
        for i in range(len(selection1['points'])):
            selected_id.append(selection1['points'][i]['customdata'])

    fig_left = get_cat_bar(df_questions, 'correct_answer', selected_id, showlegend=True)
    fig_mid = get_cat_bar(df_questions, 'model_choice', selected_id, showlegend=True)
    fig_right = get_cat_bar(df_questions, 'correct', selected_id, showlegend=True)
    fig_right_r = get_cat_bar(df_questions, 'Level', selected_id, showlegend=True)
    
    return fig_left, fig_mid, fig_right, fig_right_r

@callback(
    Output(component_id='table', component_property='data'),
    [Input("scatter-left", "selectedData")]
)
def update_table(selection1):
    if selection1 is None:
        return df_questions.to_dict('records')
    
    else:
        selected_id = []
        for i in range(len(selection1['points'])):
            selected_id.append(selection1['points'][i]['customdata'])
        return df_questions.iloc[selected_id].to_dict('records')

@callback([Output('wc_all', 'src'), 
           Output('wc_selected', 'src'), 
          ],
          [Input("scatter-left", "selectedData")],
          allow_duplicate=True)
def update_wordcloud(selection1):
    if selection1 is None:
        selected_id = None
    else:
        selected_id = []
        for i in range(len(selection1['points'])):
            selected_id.append(selection1['points'][i]['customdata'])

    return get_wordcloud(df_questions, None), get_wordcloud(df_questions, selected_id)


if __name__ == '__main__':
    app.run(debug=True)