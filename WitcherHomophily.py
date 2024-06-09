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
            data['probability'] = data['weight'] / total_weight
            # Print the probabilities
            print(f"Edge ({u}, {v}) - Weight: {data['weight']}, Probability: {data['probability']:.2f}")

# Normalize weights for each book's graph
for book, graph in graphs_by_book.items():
    print(f"\nNormalized edge probabilities for Book {book}:\n")
    normalize_weights(graph)

# Normalize weights for the combined graph
print("\nNormalized edge probabilities for Combined Graph:\n")
normalize_weights(combined_graph)

# Function to visualize a graph with edge probabilities
def visualize_graph_with_probabilities(G, title):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15)
    edge_labels = nx.get_edge_attributes(G, 'probability')
    formatted_edge_labels = {edge: f"{prob:.2f}" for edge, prob in edge_labels.items()}
    nx.draw(G, pos, with_labels=True, node_size=90, font_size=10, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels, font_color='red')
    plt.title(title, fontsize=30, fontweight='bold', color='blue', loc='center', pad=20)
    plt.show()

# Visualize each book's graph with edge probabilities
for book, graph in graphs_by_book.items():
    visualize_graph_with_probabilities(graph, f"Graph for Book {book}")

# Visualize the combined graph with edge probabilities
visualize_graph_with_probabilities(combined_graph, "Combined Graph for All Books")
