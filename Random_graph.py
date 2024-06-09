import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import powerlaw

# Load Network data
df = pd.read_csv('./Data/witcher_network.csv')
G = nx.from_pandas_edgelist(df, 'Source', 'Target', create_using=nx.Graph())  # Changed to create an undirected graph
# Find the largest strongly connected component
largest_component = max(nx.connected_components(G), key=len)  # Changed to nx.connected_components for undirected graph
G_largest = G.subgraph(largest_component)

# Check if the graph is connected
if not nx.is_connected(G_largest):
    # Take the largest connected component
    largest_cc = max(nx.connected_components(G_largest), key=len)
    graph_of_thrones = G_largest.subgraph(largest_cc).copy()

# Task 1.1: G(n,p) Model
n_p_graph = nx.erdos_renyi_graph(len(G_largest.nodes), 0.1)

# Task 1.2: G(n,m) Model
# Ensure the graph is connected
max_attempts = 100  # set a maximum number of attempts
attempt = 0
while True:
    n_m_graph = nx.erdos_renyi_graph(len(G_largest.nodes), 0.2)  # Adjust the probability as needed
    if nx.is_connected(n_m_graph):
        break

    attempt += 1
    if attempt >= max_attempts:
        print("Failed to generate a connected G(n,m) model. Exiting.")
        break

# Take the largest connected component of the generated G(n,m) model
largest_cc_n_m = max(nx.connected_components(n_m_graph), key=len)
n_m_graph = n_m_graph.subgraph(largest_cc_n_m).copy()

# Task 1.3: Configuration Model
configuration_model = nx.configuration_model(list(dict(G_largest.degree()).values()))

# Take the largest connected component of the configuration model
largest_cc_configuration_model = max(nx.connected_components(configuration_model), key=len)
configuration_model = configuration_model.subgraph(largest_cc_configuration_model).copy()

# Task 1.4: Bonus - Block Model (SBM)
# Assume you have a list of nodes in each block
blocks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
block_model = nx.stochastic_block_model(sizes=[len(block) for block in blocks], p=[[0.8, 0.2, 0.1], [0.2, 0.8, 0.1], [0.1, 0.1, 0.8]])

# Display the original and generated graphs (you can use more sophisticated visualization if needed)
plt.figure(figsize=(15, 5))

plt.subplot(1, 5, 1)
nx.draw(G_largest, with_labels=False, node_size=10)
plt.title('Original Network')

plt.subplot(1, 5, 2)
nx.draw(n_p_graph, with_labels=False, node_size=10)
plt.title('G(n,p) Model')

plt.subplot(1, 5, 3)
nx.draw(n_m_graph, with_labels=False, node_size=10)
plt.title('G(n,m) Model')

plt.subplot(1, 5, 4)
nx.draw(configuration_model, with_labels=False, node_size=10)
plt.title('Configuration Model')

plt.subplot(1, 5, 5)
nx.draw(block_model, with_labels=False, node_size=10)
plt.title('Block Model')

plt.show()

# Task 2.1: Degree Distribution
plt.figure(figsize=(12, 8))
original_degree_sequence = [d for n, d in G_largest.degree()]
n_p_degree_sequence = [d for n, d in n_p_graph.degree()]
n_m_degree_sequence = [d for n, d in n_m_graph.degree()]
config_model_degree_sequence = [d for n, d in configuration_model.degree()]
block_model_degree_sequence = [d for n, d in block_model.degree()]

plt.hist(original_degree_sequence, alpha=0.5, label='Witcher Network', bins=20)
plt.hist(n_p_degree_sequence, alpha=0.5, label='G(n,p) Model', bins=20)
plt.hist(n_m_degree_sequence, alpha=0.5, label='G(n,m) Model', bins=20)
plt.hist(config_model_degree_sequence, alpha=0.5, label='Configuration Model', bins=20)
plt.hist(block_model_degree_sequence, alpha=0.5, label='Block Model', bins=20)

plt.legend()
plt.title('Degree Distribution')
plt.show()

# Task 2.2: Giant Component
giant_component_sizes = [len(c) for c in nx.connected_components(G_largest)]
print(f'Original Giant Component Size: {max(giant_component_sizes)} nodes')

# Calculate giant component size for each model
n_p_giant_component = max(len(c) for c in nx.connected_components(n_p_graph))
n_m_giant_component = max(len(c) for c in nx.connected_components(n_m_graph))
config_model_giant_component = max(len(c) for c in nx.connected_components(configuration_model))
block_model_giant_component = max(len(c) for c in nx.connected_components(block_model))

print(f'G(n,p) Model Giant Component Size: {n_p_giant_component} nodes')
print(f'G(n,m) Model Giant Component Size: {n_m_giant_component} nodes')
print(f'Configuration Model Giant Component Size: {config_model_giant_component} nodes')
print(f'Block Model Giant Component Size: {block_model_giant_component} nodes')

# Task 2.3: Average Distance
original_avg_distance = nx.average_shortest_path_length(G_largest)
n_p_avg_distance = nx.average_shortest_path_length(n_p_graph)
n_m_avg_distance = nx.average_shortest_path_length(n_m_graph)
config_model_avg_distance = nx.average_shortest_path_length(configuration_model)
block_model_avg_distance = nx.average_shortest_path_length(block_model)

print(f'Original Average Distance: {original_avg_distance}')
print(f'G(n,p) Model Average Distance: {n_p_avg_distance}')
print(f'G(n,m) Model Average Distance: {n_m_avg_distance}')
print(f'Configuration Model Average Distance: {config_model_avg_distance}')
print(f'Block Model Average Distance: {block_model_avg_distance}')

# Task 3: Check for power-law degree distribution
degree_sequence = sorted([d for n, d in G_largest.degree()], reverse=True)
fit = powerlaw.Fit(degree_sequence, discrete=True)
alpha = fit.power_law.alpha

# Plot the degree distribution and power-law fit
plt.figure(figsize=(12, 8))
fit.plot_pdf(color='b', linestyle='--', label='Power-law fit, alpha={}'.format(round(alpha, 2)))
plt.hist(degree_sequence, bins=20, density=True, alpha=0.5, color='g', label='Degree Distribution')
plt.legend()
plt.title('Power-law Degree Distribution')
plt.show()

print(f'Power-law exponent (alpha): {alpha}')
