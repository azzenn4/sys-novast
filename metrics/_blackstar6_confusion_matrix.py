import pandas as pd
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, DistilBertConfig
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# Define label mappings
label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
reverse_label_map = {v: k for k, v in label_map.items()}  # Reverse mapping for converting labels to integers

# Load the dataset
df = pd.read_csv('MLAI_project/Lexicograph_Emotion_Detect/dataset/emotions.csv')

# Ensure the 'label' column exists
if 'label' not in df.columns:
    print("Error: 'label' column is missing from the dataset.")
    exit(1)

# Convert text labels to numeric if they're not already numeric
if df['label'].dtype == 'object':
    df['label'] = df['label'].map(reverse_label_map)

# Verify label conversion
if df['label'].isnull().any():
    print("Error: Some labels could not be mapped properly.")
    exit(1)

# Sample a smaller subset for faster debugging
sample_size = 20000  # Adjust sample size as needed
df_sampled = df.sample(n=sample_size, random_state=42)


# Split the sampled dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_sampled['text'].tolist(),
    df_sampled['label'].tolist(),
    test_size=0.2,
    random_state=42
)

model_6_path = "/home/paxi/VSC/MLAI_project/Lexicograph_Emotion_Detect/quantized_blackstar_6.pth"
tokenizer = RobertaTokenizer.from_pretrained("/home/paxi/VSC/MLAI_project/Lexicograph_Emotion_Detect/blackstar_6")
model = RobertaForSequenceClassification.from_pretrained("/home/paxi/VSC/MLAI_project/Lexicograph_Emotion_Detect/blackstar_6", num_labels=6)
model.load_state_dict(torch.load(model_6_path), strict=False)
model.eval()  # Set model to evaluation mode

# Define a function for tokenization and encoding
def tokenize_and_encode(texts, labels):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs['labels'] = torch.tensor(labels)
    return inputs

# Create datasets with labels
train_dataset = tokenize_and_encode(train_texts, train_labels)
val_dataset = tokenize_and_encode(val_texts, val_labels)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Move validation inputs to the device
val_inputs = {k: v.to(device) for k, v in val_dataset.items() if k != 'labels'}
val_labels = val_dataset['labels'].to(device)








def plot_classification_analysis(val_labels, val_inputs, model, label_map):
    # Convert labels if they're one-hot encoded
    true_labels = val_labels.argmax(dim=-1).cpu().numpy() if len(val_labels.shape) > 1 else val_labels.cpu().numpy()
    
    with torch.no_grad():
        # Get the raw logits from the model
        outputs = model(**val_inputs)
        logits = outputs.logits.cpu().numpy()
        
        # Calculate softmax probabilities
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        predictions_softmax = np.argmax(probabilities, axis=-1)
    
    # Convert label_map to list for plotting
    label_map_list = list(label_map.values())
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # First subplot: Confusion Matrix
    cm_softmax = confusion_matrix(true_labels, predictions_softmax)
    sns.heatmap(
        cm_softmax,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=label_map_list,
        yticklabels=label_map_list,
        ax=axes[0],
        square=True
    )
    axes[0].set_xlabel("Prediction")
    axes[0].set_ylabel("Truth")
    axes[0].set_title(f"Softmax [{sample_size}]")
    
    # Rotate x-axis labels for better readability
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
    
    # Second subplot: Raw Logits Heatmap
    sample_size_r = min(sample_size, logits.shape[0])  # Show up to 50 samples
    logits_subset = logits[:sample_size_r]
    
    sns.heatmap(
        logits_subset,
        annot=False,
        cmap="Oranges",
        cbar=True,
        xticklabels=label_map_list,
        yticklabels=False,
        ax=axes[1]
    )
    axes[1].set_xlabel("Classes")
    axes[1].set_ylabel("Samples")
    axes[1].set_title(f"Logits Distribution [{sample_size}]")

    # Rotate x-axis labels for better readability
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    # Add color bar labels
    for im, title in zip(axes, ['Number of Samples', 'Logit Value']):
        cbar = im.collections[0].colorbar
        cbar.set_label(title)
    
    plt.tight_layout()
    
    # Calculate and return additional metrics
    metrics = {
        'confusion_matrix': cm_softmax,
        'raw_logits_stats': {
            'mean': np.mean(logits, axis=0),
            'std': np.std(logits, axis=0),
            'min': np.min(logits, axis=0),
            'max': np.max(logits, axis=0)
        }
    }
    
    return fig, metrics

fig, metrics = plot_classification_analysis(
    val_labels=val_labels,
    val_inputs=val_inputs,
    model=model,
    label_map=label_map
)

plt.show()




