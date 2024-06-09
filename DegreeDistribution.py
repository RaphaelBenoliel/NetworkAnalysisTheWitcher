import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Data/witcher_network.csv')

# Drop the unnecessary 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in df.columns:
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

# Combine interactions across all books
combined_df = df.groupby(['Source', 'Target']).agg({'Weight': 'sum'}).reset_index()

# Create the combined graph
combined_graph = nx.Graph()

# Add edges to the combined graph
for _, row in combined_df.iterrows():
    combined_graph.add_edge(row['Source'], row['Target'], weight=row['Weight'])

# Function to visualize a graph
def visualize_graph(G, title):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15)
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=10, edge_color='gray')
    plt.title(title, fontsize=30, fontweight='bold', color='blue', loc='center', pad=20)
    plt.show()

# Visualize each book's graph
for book, graph in graphs_by_book.items():
    visualize_graph(graph, f"Graph for Book {book}")

# Visualize the combined graph
visualize_graph(combined_graph, "Combined Graph for All Books")

# Function to plot degree distribution
def plot_degree_distribution(G, title):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(max(degrees)+1), density=True, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"Charts/{title}.png")
    plt.show()

# Plot degree distribution for each book's graph
for book, graph in graphs_by_book.items():
    plot_degree_distribution(graph, f"Degree Distribution for Book {book}")

# Plot degree distribution for the combined graph
plot_degree_distribution(combined_graph, "Degree Distribution for Combined Graph")
