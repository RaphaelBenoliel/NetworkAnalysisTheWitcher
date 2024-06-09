import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# Load the dataset
df = pd.read_csv('Data/witcher_network.csv')

# Drop the unnecessary 'Unnamed: 0' column
df = df.drop(columns=['Unnamed: 0'])

# Create a dictionary to store graphs for each book
graphs_by_book = {}

# Get the unique book numbers
books = df['Book'].unique()

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


# Normalize weights for each book's graph
for book, graph in graphs_by_book.items():
    normalize_weights(graph)

# Normalize weights for the combined graph
normalize_weights(combined_graph)


# Function to perform the Independent Cascade Model
def independent_cascade(G, seed_node, iterations=10):
    total_active_nodes = 0

    for _ in range(iterations):
        active_nodes = set([seed_node])
        newly_active_nodes = set([seed_node])

        while newly_active_nodes:
            next_newly_active_nodes = set()
            for node in newly_active_nodes:
                neighbors = set(G.neighbors(node))
                inactive_neighbors = neighbors - active_nodes
                for neighbor in inactive_neighbors:
                    if random.random() <= G[node][neighbor]['probability']:
                        next_newly_active_nodes.add(neighbor)
            active_nodes.update(next_newly_active_nodes)
            newly_active_nodes = next_newly_active_nodes

        total_active_nodes += len(active_nodes)

    return total_active_nodes / iterations


# Function to run the Independent Cascade Model for each book and collect results
def run_icm_for_books(graphs_by_book, seed_nodes, iterations=10):
    all_results = {}

    for book, graph in graphs_by_book.items():
        print(f"\nRunning ICM for Book {book}")
        book_results = {}
        for seed_node in seed_nodes:
            if seed_node in graph.nodes:
                avg_active_nodes = independent_cascade(graph, seed_node, iterations)
                book_results[seed_node] = avg_active_nodes
            else:
                book_results[seed_node] = 0
        all_results[book] = book_results

    return all_results


# Function to perform the Independent Cascade Model with visualization steps
def independent_cascade_visual(G, seed_node, iterations=10):
    steps = []

    for _ in range(iterations):
        active_nodes = set([seed_node])
        newly_active_nodes = set([seed_node])
        steps.append((set(), newly_active_nodes.copy(), active_nodes.copy()))  # Initial step

        while newly_active_nodes:
            next_newly_active_nodes = set()
            for node in newly_active_nodes:
                neighbors = set(G.neighbors(node))
                inactive_neighbors = neighbors - active_nodes
                for neighbor in inactive_neighbors:
                    if random.random() <= G[node][neighbor]['probability']:
                        next_newly_active_nodes.add(neighbor)
            active_nodes.update(next_newly_active_nodes)
            steps.append((newly_active_nodes.copy(), next_newly_active_nodes.copy(), active_nodes.copy()))
            newly_active_nodes = next_newly_active_nodes

    return steps


# Function to visualize the Independent Cascade Model step by step
def visualize_icm_steps(G, steps, seed_node, book_title):
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency

    for i, (newly_active, next_newly_active, all_active) in enumerate(steps):
        plt.figure(figsize=(12, 8))

        # Node colors
        node_colors = []
        for node in G.nodes():
            if node in all_active:
                node_colors.append('green')
            elif node in newly_active:
                node_colors.append('blue')
            else:
                node_colors.append('orange')

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)

        # Draw edges
        edge_colors = []
        for u, v in G.edges():
            if (u in newly_active and v in next_newly_active) or (v in newly_active and u in next_newly_active):
                edge_colors.append('blue' if random.random() <= G[u][v]['probability'] else 'red')
            else:
                edge_colors.append('gray')

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='white')

        plt.title(f"Independent Cascade Step {i + 1}: {seed_node} (Book {book_title})")
        plt.grid(True)
        plt.savefig(f"Charts/Book_{book_title}_{seed_node}_ICM_Step_{i + 1}.png")
        plt.show()


# Number of runs and iterations per ICM
num_runs = 100
iterations = 10

# Predefined seed nodes
predefined_seed_nodes = ['Geralt', 'Vilgefortz']

# Run the Independent Cascade Model for multiple runs with a random third character
results = defaultdict(lambda: defaultdict(list))
all_steps = defaultdict(lambda: defaultdict(list))

for run in range(num_runs):
    print(f"\nRun {run + 1}/{num_runs}")

    # Select a random third character for this run
    all_nodes = set(combined_graph.nodes()) - set(predefined_seed_nodes)
    random_third_node = random.choice(list(all_nodes))
    seed_nodes = predefined_seed_nodes + [random_third_node]

    # Run the Independent Cascade Model for each book
    icm_results = run_icm_for_books(graphs_by_book, seed_nodes, iterations)

    # Run the Independent Cascade Model for the combined graph
    combined_results = run_icm_for_books({'Combined': combined_graph}, seed_nodes, iterations)

    # Collect results
    for book, book_results in icm_results.items():
        for node, active_nodes in book_results.items():
            results[book][node].append(active_nodes)

    for node, active_nodes in combined_results['Combined'].items():
        results['Combined'][node].append(active_nodes)

    # Collect steps for visualization (only one iteration for visualization purposes)
    for book, graph in graphs_by_book.items():
        for seed_node in seed_nodes:
            if seed_node in graph.nodes:
                steps = independent_cascade_visual(graph, seed_node, iterations=1)
                all_steps[book][seed_node].append(steps)

    steps_combined = independent_cascade_visual(combined_graph, random_third_node, iterations=1)
    all_steps['Combined'][random_third_node].append(steps_combined)

# Calculate average results across all runs
average_results = defaultdict(dict)

for book, book_results in results.items():
    for node, active_nodes in book_results.items():
        average_results[book][node] = sum(active_nodes) / len(active_nodes)

# Prepare data for plotting
book_labels = list(average_results.keys())
geralt_data = [average_results[book]['Geralt'] for book in book_labels]
vilgefortz_data = [average_results[book]['Vilgefortz'] for book in book_labels]
random_character_data = [average_results[book][random_third_node] for book in book_labels]

# Plotting the average results
plt.figure(figsize=(14, 7))

plt.plot(book_labels, geralt_data, label='Geralt', marker='o', color='b')
plt.plot(book_labels, vilgefortz_data, label='Vilgefortz', marker='o', color='r')
plt.plot(book_labels, random_character_data, label=f'Random Character', marker='o', color='g')

plt.xlabel('Books')
plt.ylabel('Average Number of Nodes Reached')
plt.title('Independent Cascade Model Results for Each Book and Combined Graph')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Charts/ICM_Average_Results.png")
plt.show()

# Normalize the data by the number of nodes in each graph
normalized_geralt_data = [average_results[book]['Geralt'] / graphs_by_book[book].number_of_nodes() for book in
                          book_labels[:-1]]
normalized_vilgefortz_data = [average_results[book]['Vilgefortz'] / graphs_by_book[book].number_of_nodes() for book in
                              book_labels[:-1]]
normalized_random_character_data = [average_results[book][random_third_node] / graphs_by_book[book].number_of_nodes()
                                    for book in book_labels[:-1]]

# Adding the combined graph to normalized data
normalized_geralt_data.append(average_results['Combined']['Geralt'] / num_combined_nodes)
normalized_vilgefortz_data.append(average_results['Combined']['Vilgefortz'] / num_combined_nodes)
normalized_random_character_data.append(average_results['Combined'][random_third_node] / num_combined_nodes)

# Plotting the normalized average results
plt.figure(figsize=(14, 7))

plt.plot(book_labels, normalized_geralt_data, label='Geralt', marker='o', color='b')
plt.plot(book_labels, normalized_vilgefortz_data, label='Vilgefortz', marker='o', color='r')
plt.plot(book_labels, normalized_random_character_data, label='Random Character', marker='o', color='g')

plt.xlabel('Books')
plt.ylabel('Normalized Average Number of Nodes Reached')
plt.title('Normalized Independent Cascade Model Results for Each Book and Combined Graph')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Charts/ICM_Normalized_Average_Results.png")
plt.show()

# Visualize the steps for each run (using one run's steps for visualization)
for book, book_steps in all_steps.items():
    for seed_node, steps_list in book_steps.items():
        for steps in steps_list[:1]:  # Only visualizing the first run's steps for clarity
            visualize_icm_steps(graphs_by_book.get(book, combined_graph), steps, seed_node, book)
