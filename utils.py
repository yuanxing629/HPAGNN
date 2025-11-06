import torch
import numpy as np
import networkx as nx
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
import matplotlib
from torch_geometric.utils.num_nodes import maybe_num_nodes
from textwrap import wrap


class PlotUtils():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def plot(self, graph, nodelist, figname, **kwargs):
        """ plot function for different dataset """
        if self.dataset_name.lower() == 'BA_2motifs'.lower():
            self.plot_ba2motifs(graph, nodelist, figname=figname)
        elif self.dataset_name.lower() in ['bbbp', 'mutag', 'clintox', 'mutagenicity']:
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, figname=figname)
        else:
            raise NotImplementedError

    def plot_subgraph(self, graph, nodelist, colors='#FFA500', labels=None, edge_color='gray',
                      edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                        n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        # plt.show()
        plt.close('all')

    def plot_ba2motifs(self, graph, nodelist, edgelist=None, figname=None):
        return self.plot_subgraph(graph, nodelist, edgelist=edgelist, figname=figname)

    def plot_molecule(self, graph, nodelist, x, edgelist=None, figname=None):
        # collect the text information and node color
        if self.dataset_name.lower() == 'mutag':
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            # node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
            node_color = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#E69F00', '#333333']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        elif self.dataset_name.lower() == 'mutagenicity':
            node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na',
                         11: 'K', 12: 'Li', 13: 'Ca'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            # node_color = [
            #     '#FF0000',  # 红色 - C
            #     '#00FF00',  # 绿色 - O
            #     '#0000FF',  # 蓝色 - Cl
            #     '#FFFF00',  # 黄色 - H
            #     '#FF00FF',  # 洋红 - N
            #     '#00FFFF',  # 青色 - F
            #     '#FFA500',  # 橙色 - Br
            #     '#800080',  # 紫色 - S
            #     '#A52A2A',  # 棕色 - P
            #     '#FF1493',  # 深粉红 - I
            #     '#008080',  # 深青 - Na
            #     '#FFD700',  # 金色 - K
            #     '#C0C0C0',  # 银色 - Li
            #     '#8B4513'  # 鞍棕色 - Ca
            # ]
            node_color = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2',
                          '#7F7F7F', '#BCBD22', '#17BECF', '#FFD700', '#FF1493', '#A52A2A', '#00CED1']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        elif self.dataset_name.lower() == 'bbbp' or 'clintox':
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                           for k, v in element_idxs.items()}
            node_color = ['#29A329', 'lime', '#F0EA00', 'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
            colors = [node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, nodelist, colors=colors, labels=node_labels,
                           edgelist=edgelist, edge_color='gray',
                           subgraph_edge_color='black',
                           title_sentence=None, figname=figname)
