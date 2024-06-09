import pandas as pd
import networkx as nx
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

# Print information about each graph
for book, graph in graphs_by_book.items():
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print(f"Book {book}:\nNumber of nodes: {num_nodes}\nNumber of edges: {num_edges}\n")

# Combine interactions across all books
combined_df = df.groupby(['Source', 'Target']).agg({'Weight': 'sum'}).reset_index()

# Create the combined graph
combined_graph = nx.Graph()

# Add edges to the combined graph
for _, row in combined_df.iterrows():
    combined_graph.add_edge(row['Source'], row['Target'], weight=row['Weight'])

# Print information about the combined graph
num_combined_nodes = combined_graph.number_of_nodes()
num_combined_edges = combined_graph.number_of_edges()
print(f"Combined Graph:\nNumber of nodes: {num_combined_nodes}\nNumber of edges: {num_combined_edges}\n")


# Function to normalize edge weights
def normalize_weights(G):
    for node in G.nodes():
        total_weight = sum(data['weight'] for _, _, data in G.edges(node, data=True))
        for u, v, data in G.edges(node, data=True):
            data['probability'] = data['weight'] / total_weight if total_weight > 0 else 0
            # Print the probabilities
            print(f"Edge ({u}, {v}) - Weight: {data['weight']}, Probability: {data['probability']:.2f}")


# Normalize weights for each book's graph
for book, graph in graphs_by_book.items():
    print(f"\nNormalized edge probabilities for Book {book}:\n")
    normalize_weights(graph)

# Normalize weights for the combined graph
print("\nNormalized edge probabilities for Combined Graph:\n")
normalize_weights(combined_graph)


# Function to get edge probabilities for specific nodes
def get_edge_probabilities(G, node):
    probabilities = {}
    for u, v, data in G.edges(node, data=True):
        probabilities[(u, v)] = data['probability']
    return probabilities


# Function to visualize a graph with edge probabilities and custom node colors
def visualize_graph_with_probabilities(G, title, geralt_probs, emhyr_probs):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15)

    # Set node colors
    node_colors = []
    for node in G.nodes():
        if node == 'Geralt':
            node_colors.append('red')
        elif node == 'Emhyr':
            node_colors.append('darkorange')
        else:
            node_colors.append('lightblue')

    edge_labels = nx.get_edge_attributes(G, 'probability')
    formatted_edge_labels = {edge: f"{prob:.2f}" for edge, prob in edge_labels.items()}

    nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, edge_color='gray', node_color=node_colors)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels, font_color='red')

    plt.title(title, fontsize=20, fontweight='bold', color='blue', loc='center', pad=20)

    # Display probabilities for Geralt and Emhyr var Emreis
    plt.text(1.05, 0.5, f"Geralt Probabilities:\n{geralt_probs}",
             horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(1.05, 0.3, f"Emhyr Probabilities:\n{emhyr_probs}",
             horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    plt.show()


# Visualize each book's graph with edge probabilities
for book, graph in graphs_by_book.items():
    geralt_probs = get_edge_probabilities(graph, 'Geralt')
    emhyr_probs = get_edge_probabilities(graph, 'Emhyr')
    visualize_graph_with_probabilities(graph, f"Graph for Book {book}", geralt_probs, emhyr_probs)

# Visualize the combined graph with edge probabilities
combined_geralt_probs = get_edge_probabilities(combined_graph, 'Geralt')
combined_emhyr_probs = get_edge_probabilities(combined_graph, 'Emhyr')
visualize_graph_with_probabilities(combined_graph, "Combined Graph for All Books", combined_geralt_probs,
                                   combined_emhyr_probs)


# Function to get node ranks
def get_node_rank(G, node):
    return nx.degree_centrality(G)[node]


# Collect rank data for visualization
books_list = list(books)
geralt_ranks = []
emhyr_ranks = []
geralt_prob_sums = []
emhyr_prob_sums = []

for book in books_list:
    G = graphs_by_book[book]
    geralt_rank = get_node_rank(G, 'Geralt') if 'Geralt' in G else 0
    emhyr_rank = get_node_rank(G, 'Emhyr') if 'Emhyr' in G else 0
    geralt_ranks.append(geralt_rank)
    emhyr_ranks.append(emhyr_rank)

    geralt_probs = get_edge_probabilities(G, 'Geralt')
    emhyr_probs = get_edge_probabilities(G, 'Emhyr')
    geralt_prob_sums.append(sum(geralt_probs.values()))
    emhyr_prob_sums.append(sum(emhyr_probs.values()))

# Combined graph ranks and probabilities
combined_geralt_rank = get_node_rank(combined_graph, 'Geralt')
combined_emhyr_rank = get_node_rank(combined_graph, 'Emhyr')
combined_geralt_prob_sum = sum(combined_geralt_probs.values())
combined_emhyr_prob_sum = sum(combined_emhyr_probs.values())

# Add combined ranks and probabilities to the list
books_list.append('Combined')
geralt_ranks.append(combined_geralt_rank)
emhyr_ranks.append(combined_emhyr_rank)
geralt_prob_sums.append(combined_geralt_prob_sum)
emhyr_prob_sums.append(combined_emhyr_prob_sum)

# Plot the rank distribution
plt.figure(figsize=(14, 8))
plt.plot(books_list, geralt_ranks, marker='o', label='Geralt Rank')
plt.plot(books_list, emhyr_ranks, marker='o', label='Emhyr var Emreis Rank')
plt.plot(books_list, geralt_prob_sums, marker='x', linestyle='--', label='Geralt Probability Sum')
plt.plot(books_list, emhyr_prob_sums, marker='x', linestyle='--', label='Emhyr var Emreis Probability Sum')
plt.xlabel('Book')
plt.ylabel('Rank / Probability Sum')
plt.title('Rank and Probability Distribution of Geralt and Emhyr var Emreis')
plt.legend()
plt.grid(True)
plt.show()
