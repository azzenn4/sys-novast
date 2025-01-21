import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords
def download_stopwords():
    nltk.download('stopwords')

def clean_text(text):
    """Clean a single text entry."""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove special characters, emojis, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def remove_stopwords(text, stop_words):
    """Remove stopwords from a text entry."""
    return ' '.join(word for word in text.split() if word not in stop_words)

def clean_dataset(file_path, output_path, label_filter=None):
    """Clean the dataset and save the result to a new file."""
    print("Loading dataset...")
    df = pd.read_csv(file_path)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("The dataset must have 'text' and 'label' columns.")

    print(f"Dataset loaded. {len(df)} rows.")

    # Drop duplicates
    print("Removing duplicates...")
    df = df.drop_duplicates(subset='text')

    # Clean text column
    print("Cleaning text column...")
    df['text'] = df['text'].apply(clean_text)

    # Remove stopwords
    print("Removing stopwords...")
    stop_words = set(stopwords.words('english'))
    df['text'] = df['text'].apply(lambda x: remove_stopwords(x, stop_words))

    # Remove empty text rows
    print("Removing empty or invalid text rows...")
    df = df[df['text'].str.strip() != '']

    # Apply label filter if provided
    if label_filter:
        print(f"Filtering dataset by labels: {label_filter}")
        df = df[df['label'].isin(label_filter)]

    # Reset index
    df = df.reset_index(drop=True)

    print(f"Cleaning complete. Final dataset has {len(df)} rows.")

    # Save the cleaned dataset
    print("Saving cleaned dataset...")
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    # File paths
    input_file = "dataset\suicidal.csv"  # Replace with your input file
    output_file = "dataset\clean_suicidal.csv"  # Replace with your output file

    # Labels to keep (optional)
    labels_to_keep = ["suicide", "non-suicide"]  # Example: keep only positive and negative labels

    # Ensure stopwords are downloaded
    download_stopwords()

    # Run the cleaning process
    clean_dataset(input_file, output_file, label_filter=labels_to_keep)
