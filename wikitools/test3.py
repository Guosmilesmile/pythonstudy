import networkx as nx
G=nx.Graph()
G.add_node(1)
H=nx.path_graph(10)
G.add_node(H)
print G.nodes()