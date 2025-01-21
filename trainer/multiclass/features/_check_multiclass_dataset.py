import pandas as pd
import matplotlib.pyplot as plt

# Define the mapping
emotion_to_label = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

# Load the CSV file
df = pd.read_csv("dataset/emotions.csv")

# Map the emotion labels to their names
df['emotion_label'] = df['label'].map(emotion_to_label)

# Count the occurrences of each emotion
emotion_counts = df['emotion_label'].value_counts()

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(emotion_counts.index, emotion_counts.values, color='skyblue')
plt.title("Emotion Distribution", fontsize=16)
plt.xlabel("Emotion", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
