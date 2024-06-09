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

# Function to visualize a graph
def visualize_graph(G, title):
    # Calculate degree centrality
    centrality = nx.degree_centrality(G)
    max_centrality = max(centrality.values())
    most_influential = [node for node, cent in centrality.items() if cent == max_centrality]

    print(title)
    print(f"Degree centrality for the graph: {centrality}")
    print(f"Most influential node(s) in the graph: {most_influential}")
    # Define node sizes
    node_sizes = [1000 if node in most_influential else 300 for node in G.nodes()]

    # Define node colors
    node_colors = ['red' if node in most_influential else 'skyblue' for node in G.nodes()]

    # Define node labels to emphasize the most influential nodes
    labels = {node: node if node in most_influential else '' for node in G.nodes()}

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15, iterations=50)
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=node_sizes, font_size=10, edge_color='gray',
            node_color=node_colors)
    plt.title(title, fontsize=30, fontweight='bold', color='blue', loc='center', pad=20)
    plt.show()

# Visualize each book's graph
for book, graph in graphs_by_book.items():
    visualize_graph(graph, f"Graph for Book {book}")

# Visualize the combined graph
visualize_graph(combined_graph, "Combined Graph for All Books")
