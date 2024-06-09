import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('./Data/witcher_network.csv')
short_df = pd.read_csv('./Data/witcher_small_network.csv')

df.head()

G = nx.Graph()

for _, edge in df.iterrows():
    G.add_edge(edge['Source'], edge['Target'], weight=edge['Weight'])

# nx.draw(G,with_labels=True, node_color='g')
deg_centrality = nx.degree_centrality(G)
cent = np.fromiter(deg_centrality.values(), float)
sizes = cent / np.max(cent) * 200
print('Number of nodes:', len(G.nodes))
print('Number of edges:', len(G.edges))
print('Number of connected components:', nx.number_connected_components(G))
print('Degree centrality:', deg_centrality)
# separate the connected components
components = nx.connected_components(G)
largest_component = max(components, key=len)
subgraph = G.subgraph(largest_component)
print('Number of nodes in the largest component:', len(subgraph.nodes))
print('Number of edges in the largest component:', len(subgraph.edges))

# Plot the degree distribution
degree_values = list(dict(G.degree()).values())
plt.hist(degree_values, bins=50, color='purple', edgecolor='black')
plt.title('Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.show()

plt.figure(figsize=(20,20))
nx.draw_networkx(G,with_labels=True, node_color='purple', node_size=sizes*10)
plt.show()


Short_G = nx.Graph()

for _, edge in short_df.iterrows():
    Short_G.add_edge(edge['Source'], edge['Target'], weight=edge['W eight'])

# nx.draw(Short_G,with_labels=True, node_color='g')
deg_centrality = nx.degree_centrality(Short_G)
cent = np.fromiter(deg_centrality.values(), float)
sizes = cent / np.max(cent) * 200
print('Number of nodes:', len(Short_G.nodes))
print('Number of edges:', len(Short_G.edges))
print('Number of connected components:', nx.number_connected_components(Short_G))
print('Degree centrality:', deg_centrality)

# show the graph
plt.figure(figsize=(20,20))
nx.draw_networkx(Short_G,with_labels=True, node_color='purple', node_size=sizes*10)
plt.show()

# Calculate connected components for G
components_G = list(nx.connected_components(G))
print("Connected Components for G:")
for i, component in enumerate(components_G, 1):
    print(f"Component {i}: {component}")

# Calculate connected components for Short_G
components_Short_G = list(nx.connected_components(Short_G))
print("\nConnected Components for Short_G:")
for i, component in enumerate(components_Short_G, 1):
    print(f"Component {i}: {component}")


# Calculate connected components for G
components_G = list(nx.connected_components(G))

# Get the sizes of connected components for G
component_sizes_G = [len(component) for component in components_G]

# Plot the sizes of connected components for G
plt.figure(figsize=(8, 6))
plt.bar(range(len(component_sizes_G)), component_sizes_G, color='purple')
plt.xlabel('Component Index')
plt.ylabel('Component Size')
plt.title('Connected Component Sizes for G')
plt.xticks(range(len(component_sizes_G)), [f'Component {i+1}' for i in range(len(component_sizes_G))])
plt.show()

# Calculate connected components for Short_G
components_Short_G = list(nx.connected_components(Short_G))

# Get the sizes of connected components for Short_G
component_sizes_Short_G = [len(component) for component in components_Short_G]

# Plot the sizes of connected components for Short_G
plt.figure(figsize=(8, 6))
plt.bar(range(len(component_sizes_Short_G)), component_sizes_Short_G, color='purple')
plt.xlabel('Component Index')
plt.ylabel('Component Size')
plt.title('Connected Component Sizes for Short_G')
plt.xticks(range(len(component_sizes_Short_G)), [f'Component {i+1}' for i in range(len(component_sizes_Short_G))])
plt.show()
