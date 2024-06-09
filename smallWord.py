import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Data/witcher_network.csv')

# Drop the unnecessary 'Unnamed: 0' column
df = df.drop(columns=['Unnamed: 0'])

# Create a dictionary to store graphs for each book
graphs_by_book = {}

# Get the unique book numbers
books = df['Book'].unique()  # Ensure column name is 'Book'

# Create a graph for each book
for book in books:
    # Filter the DataFrame for the current book
    book_df = df[df['Book'] == book]

    # Create the graph
    G = nx.Graph()

    # Add edges to the graph
    for _, row in book_df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    # Store the graph in the dictionary
    graphs_by_book[book] = G


# Function to analyze the network properties
def analyze_network(graph):
    # Size of the network
    num_nodes = graph.number_of_nodes()

    # Largest connected component
    largest_cc = max(nx.connected_components(graph), key=len)
    lcc = graph.subgraph(largest_cc).copy()
    lcc_size = len(largest_cc)

    # Average degree
    avg_degree = sum(dict(lcc.degree()).values()) / lcc_size

    # Short distance distribution
    path_lengths = dict(nx.all_pairs_shortest_path_length(lcc))
    all_lengths = [length for target_dict in path_lengths.values() for length in target_dict.values()]

    return num_nodes, lcc_size, avg_degree, all_lengths, lcc


# Function to create a random graph with the same number of nodes and edges
def create_random_graph(num_nodes, num_edges):
    p = num_edges / (num_nodes * (num_nodes - 1) / 2)
    random_graph = nx.erdos_renyi_graph(num_nodes, p)
    return random_graph


# Analyze and plot results for each book
combined_lengths = []
combined_random_lengths = []
combined_degrees = []

for book, graph in graphs_by_book.items():
    num_nodes, lcc_size, avg_degree, all_lengths, lcc = analyze_network(graph)
    num_edges = graph.number_of_edges()

    print(f"Book {book} - Total Nodes: {num_nodes}, Largest Connected Component Size: {lcc_size}, Average Degree: {avg_degree}")

    combined_lengths.extend(all_lengths)
    combined_degrees.extend([d for n, d in lcc.degree()])

    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=range(1, max(all_lengths) + 1), density=True, alpha=0.6, color='purple')
    plt.title(f'Short Distance Distribution (Histogram) - Book {book}')
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(sorted(all_lengths), 'g-')
    plt.title(f'Short Distance Distribution (Linear) - Book {book}')
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Number of Pairs')
    plt.grid(True)
    plt.show()

    # Calculate and print median and mean of the distances
    median_distance = np.median(all_lengths)
    mean_distance = np.mean(all_lengths)
    print(f"Book {book} - Median Distance: {median_distance}, Mean Distance: {mean_distance}")

    # Create and analyze a random graph for comparison
    random_graph = create_random_graph(num_nodes, num_edges)
    _, random_lcc_size, random_avg_degree, random_lengths, _ = analyze_network(random_graph)

    combined_random_lengths.extend(random_lengths)

    print(f"Book {book} - Random Graph: Largest Connected Component Size: {random_lcc_size}, Average Degree: {random_avg_degree}")

    plt.figure(figsize=(10, 6))
    plt.hist(random_lengths, bins=range(1, max(random_lengths) + 1), density=True, alpha=0.6, color='orange')
    plt.title(f'Random Graph Short Distance Distribution (Histogram) - Book {book}')
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(sorted(random_lengths), 'r-')
    plt.title(f'Random Graph Short Distance Distribution (Linear) - Book {book}')
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Number of Pairs')
    plt.grid(True)
    plt.show()

    # Calculate and print median and mean of the random graph distances
    random_median_distance = np.median(random_lengths)
    random_mean_distance = np.mean(random_lengths)
    print(f"Book {book} - Random Graph: Median Distance: {random_median_distance}, Mean Distance: {random_mean_distance}")

# Plot combined short distance distribution for the original graphs
plt.figure(figsize=(10, 6))
plt.hist(combined_lengths, bins=range(1, max(combined_lengths) + 1), density=True, alpha=0.6, color='b')
plt.title('Combined Short Distance Distribution (Histogram) - Original Graphs')
plt.xlabel('Shortest Path Length')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sorted(combined_lengths), 'b-')
plt.title('Combined Short Distance Distribution (Linear) - Original Graphs')
plt.xlabel('Shortest Path Length')
plt.ylabel('Number of Pairs')
plt.grid(True)
plt.show()

# Calculate and print combined median and mean of the distances for the original graphs
combined_median_distance = np.median(combined_lengths)
combined_mean_distance = np.mean(combined_lengths)
print(f"Combined - Original Graphs: Median Distance: {combined_median_distance}, Mean Distance: {combined_mean_distance}")

# Plot combined short distance distribution for the random graphs
plt.figure(figsize=(10, 6))
plt.hist(combined_random_lengths, bins=range(1, max(combined_random_lengths) + 1), density=True, alpha=0.6, color='g')
plt.title('Combined Short Distance Distribution (Histogram) - Random Graphs')
plt.xlabel('Shortest Path Length')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sorted(combined_random_lengths), 'g-')
plt.title('Combined Short Distance Distribution (Linear) - Random Graphs')
plt.xlabel('Shortest Path Length')
plt.ylabel('Number of Pairs')
plt.grid(True)
plt.show()

# Calculate and print combined median and mean of the distances for the random graphs
combined_random_median_distance = np.median(combined_random_lengths)
combined_random_mean_distance = np.mean(combined_random_lengths)
print(f"Combined - Random Graphs: Median Distance: {combined_random_median_distance}, Mean Distance: {combined_random_mean_distance}")

# Plot the degree distribution for the original combined graph
plt.figure(figsize=(10, 6))
plt.hist(combined_degrees, bins=range(1, max(combined_degrees) + 1), density=True, alpha=0.6, color='c')
plt.title('Combined Degree Distribution - Original Graphs')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sorted(combined_degrees), 'c-')
plt.title('Combined Degree Distribution (Linear) - Original Graphs')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.grid(True)
plt.show()

# Calculate and print the combined average degree for the original graphs
combined_avg_degree = np.mean(combined_degrees)
print(f"Combined - Original Graphs: Average Degree: {combined_avg_degree}")
