import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the dataset
df = pd.read_csv('./Data/witcher_network.csv')

# Create an undirected graph from the dataset
G = nx.from_pandas_edgelist(df, 'Source', 'Target', edge_attr=['Weight'])

# Find the largest connected component
largest_component = max(nx.connected_components(G), key=len)
G_largest = G.subgraph(largest_component)

# Calculate the number of nodes and edges in the largest component
num_nodes_largest = len(G_largest.nodes)
num_edges_largest = len(G_largest.edges)

print(f"Number of nodes in the largest connected component: {num_nodes_largest}")
print(f"Number of edges in the largest connected component: {num_edges_largest}")

# Draw the largest connected component
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G_largest)
nx.draw(G_largest, pos, with_labels=True, node_size=200, node_color='purple', font_size=10, font_weight='bold')
plt.title('Largest Connected Component of The Witcher Character Relationships')
plt.show()

# Check if the graph is directed
is_directed = nx.is_directed(G_largest)
print(f"The graph is directed: {is_directed}")

# Check if the graph is a multigraph
is_multigraph = isinstance(G_largest, nx.MultiGraph) or isinstance(G_largest, nx.MultiDiGraph)
print(f"The graph is a multigraph: {is_multigraph}")

# Number of tie elements (assuming each tie element is an edge with unique source and target)
num_tie_elements = num_edges_largest
print(f"Number of tie elements in the largest connected component: {num_tie_elements}")

# Calculate the diameter of the largest connected component
if num_nodes_largest > 1:
    diameter_largest = nx.diameter(G_largest)
    print(f"Diameter of the largest connected component: {diameter_largest}")
else:
    print("The largest connected component has only one node.")

# Check if the network supports the small-world feature
is_small_world = nx.average_clustering(G_largest) / nx.average_shortest_path_length(G_largest) > 1
print(f"The network supports the small-world feature: {is_small_world}")

# Calculate degree centrality for the largest connected component
degree_centrality_largest = nx.degree_centrality(G_largest)
print("Degree Centrality for the largest connected component:")
for node, centrality in degree_centrality_largest.items():
    print(f"{node}: {centrality}")

# Calculate betweenness centrality for the largest connected component
betweenness_centrality_largest = nx.betweenness_centrality(G_largest)
print("\nBetweenness Centrality for the largest connected component:")
for node, centrality in betweenness_centrality_largest.items():
    print(f"{node}: {centrality}")

# Calculate closeness centrality for the largest connected component
closeness_centrality_largest = nx.closeness_centrality(G_largest)
print("\nCloseness Centrality for the largest connected component:")
for node, centrality in closeness_centrality_largest.items():
    print(f"{node}: {centrality}")

# show chart of closeness centrality for the largest connected component
plt.figure(figsize=(12, 8))
plt.bar(closeness_centrality_largest.keys(), closeness_centrality_largest.values())
plt.xlabel('Nodes')
plt.ylabel('Closeness Centrality')
plt.title('Closeness Centrality for the largest connected component')
plt.xticks(rotation='vertical', fontsize=2)
plt.show()


# show chart of degree centrality for the largest connected component
plt.figure(figsize=(12, 8))
# how to make the bar chart key vertical for better readability of the node names?
plt.bar(degree_centrality_largest.keys(), degree_centrality_largest.values())
plt.xlabel('Nodes')
plt.ylabel('Degree Centrality')
plt.title('Degree Centrality for the largest connected component')
plt.xticks(rotation='vertical', fontsize=2)
plt.show()



# show chart of betweenness centrality for the largest connected component
plt.figure(figsize=(12, 8))
plt.bar(betweenness_centrality_largest.keys(), betweenness_centrality_largest.values())
plt.xlabel('Nodes')
plt.ylabel('Betweenness Centrality')
plt.title('Betweenness Centrality for the largest connected component')
plt.xticks(rotation='vertical', fontsize=2)
plt.show()


# Rank nodes by degree centrality in the largest connected component
degree_rank_largest = sorted(degree_centrality_largest.items(), key=lambda x: x[1], reverse=True)
print("Degree Centrality Rank for the largest connected component:")
for rank, (node, centrality) in enumerate(degree_rank_largest, start=1):
    print(f"{rank}. {node}: {centrality}")

# Calculate edge betweenness centrality
edge_betweenness = nx.edge_betweenness_centrality(G_largest)

# Order edges based on strong to weak edge weights
edges_strong_to_weak = sorted(G_largest.edges(data=True), key=lambda x: x[2]['Weight'], reverse=True)

# Order edges based on weak to strong edge weights
edges_weak_to_strong = sorted(G_largest.edges(data=True), key=lambda x: x[2]['Weight'])

# Order edges based on betweenness centrality
edges_betweenness_order = sorted(G_largest.edges(data=True), key=lambda x: edge_betweenness[(x[0], x[1])], reverse=True)

# Calculate giant component sizes after edge removal
def calculate_giant_component_sizes(graph, edges_order):
    component_sizes = []
    for edge in edges_order:
        graph_removed = graph.copy()
        graph_removed.remove_edge(*edge[:2])
        components = nx.connected_components(graph_removed)
        giant_component_size = max(len(comp) for comp in components)
        component_sizes.append(giant_component_size)
    return component_sizes

giant_component_sizes_strong_to_weak = calculate_giant_component_sizes(G_largest, edges_strong_to_weak)
giant_component_sizes_weak_to_strong = calculate_giant_component_sizes(G_largest, edges_weak_to_strong)
giant_component_sizes_betweenness_order = calculate_giant_component_sizes(G_largest, edges_betweenness_order)

# Plotting
plt.figure(figsize=(12, 8))

plt.plot(range(num_edges_largest), giant_component_sizes_strong_to_weak, label='Strong to Weak Edge Order', color='red')
plt.plot(range(num_edges_largest), giant_component_sizes_weak_to_strong, label='Weak to Strong Edge Order', color='green')
plt.plot(range(num_edges_largest), giant_component_sizes_betweenness_order, label='Betweenness Centrality Order', color='blue')

plt.xlabel('Number of Edges Removed')
plt.ylabel('Giant Component Size')
plt.title('Edge Removal and Giant Component Size for the largest connected component')
plt.legend()
plt.show()
# Calculate neighborhood overlap as a function of weight for the largest connected component
neighborhood_overlap_largest = defaultdict(list)
for edge in G_largest.edges(data=True):
    source_neighbors = set(G_largest.neighbors(edge[0]))
    target_neighbors = set(G_largest.neighbors(edge[1]))
    overlap = len(source_neighbors.intersection(target_neighbors))
    neighborhood_overlap_largest[edge[2]['Weight']].append(overlap)

# Plotting
plt.figure(figsize=(8, 6))
for weight, overlap_list in neighborhood_overlap_largest.items():
    plt.scatter([weight] * len(overlap_list), overlap_list, label=f'Weight {weight}')

plt.xlabel('Weight')
plt.ylabel('Neighborhood Overlap')
plt.legend()
plt.title('Neighborhood Overlap as Function of Weight for the largest connected component')
plt.show()


