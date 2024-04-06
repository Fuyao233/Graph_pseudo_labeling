# 4.4 数据可视化
from utils import load_dataset, prepocessing, DataLoader_folder
import torch 
import numpy as np
import os 
import networkx as nx 
import plotly.graph_objects as go
from plotly.offline import plot

# 观察不同代节点划分
def visulization_split(graph_name, layout, state_dic, save_name):
    print(save_name)
    graph = prepocessing(load_dataset(graph_name))
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(graph.edge_index.T.numpy())

    pos = layout
    # pos = nx.random_layout(G, seed=42)

    color = state_dic["edge_pseudolabel"].numpy()
    part_indices = torch.zeros_like(state_dic['train_index'], dtype=int)
    part_indices[state_dic['train_index_A']] = 0
    part_indices[state_dic['train_index_B']] = 1
    part_indices[state_dic['val_index']] = 2
    part_indices[state_dic['test_index']] = 3
    part_indices[state_dic['unlabeled_index']] = 4

    # TODO:考虑pseudolabel, 完善text
    part_list = ['train_A', 'train_B', 'val', 'test', 'unlabeled']
    
    pos_array = np.array(list(pos.values()))
    x_nodes = pos_array[:, 0]  
    y_nodes = pos_array[:, 1] 

    x_edges = []
    y_edges = []
    for edge in G.edges():
        x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])

    edge_trace = go.Scatter(x=x_edges, y=y_edges, line=dict(width=0.5, color='#ffffff'), hoverinfo='none', mode='lines')
    
    edge_pseudolabel = state_dic['edge_pseudolabel'].numpy()
    y_label = graph.y.numpy()
    
    # 分离出加粗边框的节点和不加粗的节点
    bold_border_indices = [i for i, (pseudo, true) in enumerate(zip(edge_pseudolabel, y_label)) if pseudo != true]
    normal_border_indices = [i for i, (pseudo, true) in enumerate(zip(edge_pseudolabel, y_label)) if pseudo == true]
    # 不加粗边框的节点
    node_trace_normal = go.Scatter(x=x_nodes[normal_border_indices], y=y_nodes[normal_border_indices], mode='markers', hoverinfo='text',
                                   text=[f'prediction {edge_pseudolabel[n]}<br>confidence {state_dic["label_confidence"][n] if state_dic["label_confidence"] is not None else "Nan"}<br>label {y_label[n]}<br>{part_list[part_indices[n]]}' for n in normal_border_indices],
                                   marker=dict(color=[color[n] for n in normal_border_indices], colorscale='Rainbow', showscale=True, line=dict(width=0.5, color='#ffffff')))
    
    # 加粗边框的节点
    node_trace_bold = go.Scatter(x=x_nodes[bold_border_indices], y=y_nodes[bold_border_indices], mode='markers', hoverinfo='text',
                                 text=[f'prediction {edge_pseudolabel[n]}<br>confidence {state_dic["label_confidence"][n] if state_dic["label_confidence"] is not None else "Nan"}<br>label {y_label[n]}<br>{part_list[color[n]]}' for n in bold_border_indices],
                                 marker=dict(color=[color[n] for n in bold_border_indices], colorscale='Rainbow', showscale=True, line=dict(width=2, color='#ffffff')))

    fig = go.Figure(data=[edge_trace, node_trace_normal, node_trace_bold], layout=go.Layout(showlegend=False, hovermode='closest',
                                                                                            paper_bgcolor='black',  plot_bgcolor='black',  # 设置图表主背景为黑色
                                                                                           margin=dict(b=0, l=0, r=0, t=0),
                                                                                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                                                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    fig.write_html(os.path.join('htmls', f'{save_name}.html'))
    fig.write_image(os.path.join('imgs', f'{save_name}.png'))

def draw_ground_truth(graph_name):
    graph = prepocessing(load_dataset(graph_name))
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(graph.edge_index.T.numpy())
    pos = nx.spring_layout(G, seed=42)
    
    color = graph.y.numpy()
    pos_array = np.array(list(pos.values()))
    x_nodes = pos_array[:, 0]  
    y_nodes = pos_array[:, 1] 

    x_edges = []
    y_edges = []
    for edge in G.edges():
        x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])

    edge_trace = go.Scatter(x=x_edges, y=y_edges, line=dict(width=0.5, color='#ffffff'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(x=x_nodes, y=y_nodes, mode='markers', hoverinfo='text', 
                            marker=dict(color=color, colorscale='Rainbow', showscale=True, line=dict(width=0.5, color='#00FF00')))
    
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest',
                                                                                            paper_bgcolor='black',  plot_bgcolor='black',  # 设置图表主背景为黑色
                                                                                           margin=dict(b=0, l=0, r=0, t=0),
                                                                                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                                                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    fig.write_html(os.path.join('htmls', f'{graph_name}.html'))
    fig.write_image(os.path.join('imgs', f'{graph_name}.png'))
    
    return pos

dataset_name_list = ["yelp-chi", "chameleon", "squirrel", "texas", "cornell", "wisconsin"]
# dataset_name_list = ["texas"]
dataset_name_list = list(reversed(dataset_name_list))
for name in dataset_name_list:
    print(name)
    loader = DataLoader_folder(os.path.join('utils_data', f'ourModel_{name}_0_A_B_random_soft'))
    state_record_list = loader.state_record_list
    pos = draw_ground_truth(name)
    [visulization_split(name, pos, state,  f'{name}_{i}_random') for i, state in enumerate(state_record_list)]
    