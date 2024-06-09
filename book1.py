import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Data/witcher_network.csv')

# Drop the unnecessary 'Unnamed: 0' column
df = df.drop(columns=['Unnamed: 0'])
for book in df['Book'].unique():
    book_df = df[df['Book'] == book]
    characters = ['Geralt', 'Vilgefortz']
    interactions_df = book_df[((book_df['Source'].isin(characters)) & (book_df['Target'].isin(characters)))]
    if interactions_df.empty:
        print(f"No interactions found between {characters[0]} and {characters[1]} in Book {book}")
    else:
        print(f"Interactions between {characters[0]} and {characters[1]} in Book {book}:")
        print(interactions_df)


# Filter the DataFrame for the first book
# book_1_df = df[df['Book'] == 1]

# Filter interactions between Geralt and Foltest



# # Print filtered interactions
# print(interactions_df)
#
# # Create a distribution model for the interaction weights
# if interactions_df is not None:
#     plt.figure(figsize=(10, 6))
#     plt.hist(interactions_df['Weight'], bins=range(1, max(interactions_df['Weight']) + 1), edgecolor='black')
#     plt.title('Distribution of Interaction Weights between Geralt and Foltest in Book 1')
#     plt.xlabel('Interaction Weight')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.show()
# else:
#     print("No interactions found for the specified characters in the" + bookNumber + "book.")




# seed_nodes = ['Geralt', 'Vilgefortz', random.choice(list(combined_graph.nodes))]