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


class GygraphAnalysis(object):
	def __init__(self):
		self.G = nx.Graph()
		self.min_node_size = 5
		self.max_node_size = 10

	def importData(self,data=None):
		self.clear_nodes()
		self.G = nx.star_graph(12)
		self.degree = nx.degree(self.G)
		self.max_degree_value = max(self.degree.values())
		self.max_degree_value_floated = float(self.max_degree_value)

	#clear all the node an edges
	def clear_nodes(self):
		self.G.clear()

	def node_size(self,n):
		d = self.degree[n]


g = GygraphAnalysis()
g.importData()
print g.G.edges()

# G=nx.star_graph(12)
# pos=nx.spring_layout(G)
# #G.add_edge(1,2)
# COLORS = [
#         '#F20010', '#FC4292', '#B94C4A', '#D19D39', '#A79D0F', '#A65FCC',
#         '#9470E7', '#19CD1D', '#1DDF9A', '#52A79F', '#24B5D9', '#2080DC',
#     ]
# nx.draw_networkx_nodes(G,pos,alpha = 0.8,node_color=COLORS)
# nx.draw_networkx_edges(G,pos,alpha=0.1,width=4,node_color=COLORS)
#nx.draw(G,pos,node_color='#A0CBE2',edge_color=colors,width=4,edge_cmap=plt.cm.Blues,with_labels=True)
#plt.savefig("edge_colormap.png") # save as png
plt.show()