import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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

# Combine all book graphs into one graph
combined_graph = nx.Graph()
for graph in graphs_by_book.values():
    combined_graph = nx.compose(combined_graph, graph)

# Function to calculate degree distribution and fit power law
def calculate_power_law(graph, degree_threshold=2):
    degrees = [degree for node, degree in graph.degree() if degree > degree_threshold]
    degree_counts = np.bincount(degrees)
    degree_probabilities = degree_counts / np.sum(degree_counts)

    # Filter out zero probabilities for log-log plot
    degrees_nonzero = np.nonzero(degree_probabilities)[0]
    probabilities_nonzero = degree_probabilities[degrees_nonzero]

    # Log-log values
    log_degrees = np.log(degrees_nonzero)
    log_probabilities = np.log(probabilities_nonzero)

    # Fit power law: log(P(k)) = -beta * log(k) + c
    def power_law_fit(k, beta, c):
        return -beta * np.log(k) + c

    popt, _ = curve_fit(power_law_fit, degrees_nonzero, log_probabilities)

    beta, c = popt

    return beta, c, degrees_nonzero, probabilities_nonzero

# Function to visualize power law fit
def visualize_power_law(graph, title, degree_threshold=2):
    beta, c, degrees, probabilities = calculate_power_law(graph, degree_threshold)

    plt.figure(figsize=(10, 6))
    plt.loglog(degrees, probabilities, 'bo', markersize=8, label='Degree Distribution')
    plt.title(f'Power Law Fit for {title}')
    plt.xlabel('Degree (k)')
    plt.ylabel('Probability (P(k))')

    fit_line = np.exp(-beta * np.log(degrees) + c)
    plt.loglog(degrees, fit_line, 'r-', linewidth=2, label=f'Power Law Fit (Beta: {beta:.2f}, C: {np.exp(c):.2f})')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    # plt.savefig(f'Charts/PowerLaw_{title}.png')
    plt.show()

    print(f"{title} - Beta: {beta}, C: {np.exp(c)}")

# Calculate and visualize power law for each book
for book, graph in graphs_by_book.items():
    visualize_power_law(graph, f'Book {book}', degree_threshold=2)

# Calculate and visualize power law for the combined graph
visualize_power_law(combined_graph, 'Combined Books', degree_threshold=2)
