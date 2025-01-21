import pandas as pd
import matplotlib.pyplot as plt

# Define the mapping for binary classification
binary_label_to_text = {0: 'Not Depressed', 1: 'Depressed'}

# Load the CSV file
df = pd.read_csv("dataset/depression_1.csv")

# Map the binary labels to descriptive text
df['label_text'] = df['label'].map(binary_label_to_text)

# Count the occurrences of each binary label
binary_counts = df['label_text'].value_counts()

# Plotting
plt.figure(figsize=(6, 5))
plt.bar(binary_counts.index, binary_counts.values, color=['skyblue', 'salmon'])
plt.title("Depression Classification Distribution", fontsize=16)
plt.xlabel("Classification", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.tight_layout()

# Show the plot
plt.show()
