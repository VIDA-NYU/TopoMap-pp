import numpy as np

from .utils import get_hull

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

def plot_topomap_comparison_highlight(proj_original, proj_new, 
                                      components_to_highligth, df_comp,
                                      hiertopomap=None,
                                      colors=[]
                                      ):
    

    highligth = np.zeros(shape=proj_original.shape[0])

    for i, comp in enumerate(components_to_highligth):
        highligth[df_comp.loc[comp]['points']] = i+1
    
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'xy'},
                                {'type': 'xy'}]],
                        subplot_titles=('Original Projection - TopoMap',
                                        'New Projection - Hierarchical TopoMap'),
                        horizontal_spacing = 0.02)
    
    highligth_values = np.unique(highligth)
    l = len(np.unique(highligth))
    if not 0 in highligth_values:
        l += 1

    if len(colors)==0:
        colors = px.colors.qualitative.T10

    num_colors = (len(components_to_highligth)//10)+1
    colors = num_colors*colors

    for i in range(l):
        if i==0:
            if not 0 in highligth_values:
                continue
            name = 'Other points'
        else:
            name = f'Component {components_to_highligth[i-1]}'

        fig.add_trace(
            go.Scatter(x=proj_original[highligth==i,0], 
                    y=proj_original[highligth==i,1],
                    mode='markers',
                    #opacity=0.5,
                    marker=dict(
                        color=colors[i],
                        size=2,
                    ),
                    name=str(name),
                    legendgroup=str(name),
                    ),
            row=1, col=1
        )

    if not hiertopomap is None:
        alphas = []
        for c in hiertopomap.components_to_scale:
            alphas.append(hiertopomap.components_info[c]['alpha'])

        #min_alpha = min(alphas)
        #range_alpha = max(alphas)-min_alpha
        max_alpha = max(alphas)
        max_color = px.colors.sample_colorscale('Greys', [1])[0]

        for j in range(len(hiertopomap.components_info)):
            comp_ids = hiertopomap.components_info[j]['points']
            if 'hull' in hiertopomap.components_info[j].keys():
                hull = hiertopomap.components_info[j]['hull']
                points_ids = [comp_ids[i] for i in hull.vertices]
                points = list(hiertopomap.projections[points_ids,:])
                points.append(points[0])
                xs, ys = zip(*points)

                #alpha_scaled = (hiertopomap.components_info[j]['alpha']-min_alpha)/range_alpha
                alpha_scaled = (hiertopomap.components_info[j]['alpha']-1)/(1.1*max_alpha-1)
                hull_color = sample_colorscale('Greys', [alpha_scaled])[0]

                fig.add_trace(go.Scatter(x=xs, y=ys,
                                fill='toself', 
                                fillcolor = hull_color,
                                line_color=max_color,
                                opacity=0.5,
                                line_width=1,
                                text=f'Component {j}',
                                name='Components', legendgroup='Components',
                                showlegend=False,
                                ),
                            row=1, col=2)
                
    for i in range(l):
        if i==0:
            if not 0 in highligth_values:
                continue
            name = 'Other points'
        else:
            name = f'Component {components_to_highligth[i-1]}'

        fig.add_trace(
            go.Scatter(x=proj_new[highligth==i,0], 
                    y=proj_new[highligth==i,1],
                    mode='markers',
                    marker=dict(
                        color=colors[i],
                        size=2,
                    ),
                    name=name,
                    legendgroup=name,
                    showlegend=False
                    ),
            row=1, col=2
        )

    fig.update_layout(margin = dict(t=75, l=25, r=25, b=25),
            height=500,
            width=1000,
            legend= {'itemsizing': 'constant',
                     'title': 'Components'},
            xaxis=dict(showticklabels=False,
                       showline=True, linewidth=1, linecolor='black', mirror=True), 
            yaxis=dict(showticklabels=False,
                       showline=True, linewidth=1, linecolor='black', mirror=True),
            xaxis2=dict(showticklabels=False,
                        showline=True, linewidth=1, linecolor='black', mirror=True), 
            yaxis2=dict(showticklabels=False,
                        showline=True, linewidth=1, linecolor='black', mirror=True),
            plot_bgcolor = "white"
            )

    return fig

def plot_projections_discrete_feature(projections,
                                    df_data,
                                    column_color,
                                    column_values=[],
                                    colors=[],
                                    hiertopomap=None,
                                    legend_title='',
                                    low_opacity=False,
                                    show_hulls=True,
                                    topomap=False
                                    ):
    if len(column_values) == 0:
        column_values = list(df_data[column_color].unique())
    if len(colors)==0:
        colors = px.colors.qualitative.T10
    if legend_title=='':
        legend_title=column_color

    fig = go.Figure()

    if show_hulls:
        for c in hiertopomap.components_to_scale:
            comp_ids = hiertopomap.components_info[c]['points']

            if topomap:
                hull = hiertopomap.components_info[c]['hull']
                points = projections[comp_ids]
                hull = get_hull(points)
                points_hull = [points[i] for i in hull.vertices]
                points_hull.append(points_hull[0])
                xs, ys = zip(*points_hull)

            else:
                hull = hiertopomap.components_info[c]['hull']
                points_ids = [comp_ids[i] for i in hull.vertices]
                points = list(hiertopomap.projections[points_ids,:])
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
                        height=500,
                        width=550,
                        legend= {'itemsizing': 'constant',
                        'title': legend_title},
                        plot_bgcolor = "white",
                        xaxis=dict(showticklabels=False,
                        showline=True, linewidth=1, linecolor='black', mirror=True), 
                        yaxis=dict(showticklabels=False,
                        showline=True, linewidth=1, linecolor='black', mirror=True),
                    )
    
    return fig

def plot_hierarchical_treemap(df_comp, color='died_at'):
    fig = go.Figure(go.Treemap(
            labels=df_comp['id'],
            parents=df_comp['parent'],
            values=df_comp['size'],
            branchvalues='total',
            marker=dict(
                colors=df_comp[color],
                colorscale='Teal',
                showscale=True,
                colorbar=dict(
                    title='Persistence',
                    titleside='right',
                    thickness=15,
                    x=1
                )),
            hovertemplate='<b>Component #%{label} </b> <br> Points: %{value}<br> Persistence: %{color:.2f}<br> Parent: #%{parent}',
            tiling=dict(packing='squarify')
            ),
        )

    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25),
                    title='TopoTree',
                    height=500,
                    width=800
                    )
    
    return fig

def plot_icicle(df_comp, color='died_at', color_title='Persistence'):
    fig = go.Figure(go.Icicle(
            labels=df_comp['id'],
            parents=df_comp['parent'],
            values=df_comp['size'],
            marker=dict(
                colors=df_comp[color],
                colorscale='Teal',
                showscale=True,
                colorbar=dict(
                    title=color_title,
                    titleside='right',
                    thickness=15,
                    x=1
                )),
            root_color="lightgrey",
            hovertemplate='<b>Component #%{label} </b> <br> Points: %{value}<br> Persistence: %{color:.2f}<br> Parent: #%{parent}',
            tiling = dict(orientation='v')
            ),
        )

    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25),
                    title='Icicle',
                    height=500,
                    width=800
                    )
    
    return fig
