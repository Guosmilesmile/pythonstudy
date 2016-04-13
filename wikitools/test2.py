#!/usr/bin/env python
"""
Draw a graph with matplotlib, color edges.
You must have matplotlib>=87.7 for this to work.
"""
__author__ = """Aric Hagberg (hagberg@lanl.gov)"""
try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx

G=nx.star_graph(20)
pos=nx.spring_layout(G)
colors=range(21)
G.add_edge(1,2)
print G.edges()
nx.draw(G,pos,node_color='#A0CBE2',edge_color=colors,width=4,edge_cmap=plt.cm.Blues,with_labels=False)
nx.draw_networkx_labels(G,pos=nx.spring_layout(G))
plt.savefig("edge_colormap.png") # save as png
plt.show()