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
import random

class GygraphAnalysis(object):
	def __init__(self):
		self.G = nx.Graph()
		self.min_node_size = 35
		self.max_node_size = 50

	def importData(self,data=None):
		self.clear_nodes()
		self.G = nx.star_graph(12)
		self.degree = nx.degree(self.G)
		self.max_degree_value = max(self.degree.values())
		self.max_degree_value_floated = float(self.max_degree_value)
		self.dict = {}
		for item in self.G.nodes():
			self.dict[item] = "k"
	#clear all the node an edges
	def clear_nodes(self):
		self.G.clear()

	#the degree of node divided by the maximum value of nodes, replace the result with minimum node size if it is smaller.  then the results is assigned to the size of the node.
	def get_one_node_size(self,n):
		d = self.degree[n]
		d = d / self.max_degree_value_floated * self.max_node_size
		if d < self.min_node_size:
			d = self.min_node_size
		return d
	
	def get_node_size(self,nodes):
		return [self.get_one_node_size(n) for n in nodes]

	def get_node_color(self, nodes):
		return [self.one_node_color(n) for n in nodes]

	def one_node_color(self, n):
		d = self.degree[n]
		if d > self.max_degree_value / 2:
			_range = [0.5, 0.8]
		else:
			_range = [0.8, 1.0]
		_make = lambda: random.uniform(*_range)
		_love = _make
		_ohyes = _make
		return (_make(), _love(), _ohyes())

	def save(self, f='result.png', it=55):
		pos = nx.spring_layout(self.G, iterations=it)
        
		nx.draw_networkx_edges(self.G, pos, alpha=0.1)
        
		nx.draw_networkx_nodes(
			self.G,
			pos,
			node_size = self.get_node_size(self.G.nodes()),
			node_color = self.get_node_color(self.G.nodes()),
			alpha = 0.8,
		)

		nx.draw_networkx_labels(
			self.G,
			pos,
			self.dict,
			font_size=5,
			alpha = 0.8,
		)
		plt.axis('off')
		plt.savefig(f, dpi=200)


g = GygraphAnalysis()
g.importData()
g.save()

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
#plt.show()