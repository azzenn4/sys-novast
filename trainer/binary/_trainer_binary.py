import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    TrainerCallback, 
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import os
import logging

'''

    Comment for more context found in ../multiclass/_trainer_multiclass.py
    
'''

# uncomment below if using wayland compositor
# os.environ['QT_QPA_PLATFORM'] = 'xcb'
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA current device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print(f"CUDA is not available: {torch.cuda.is_available()}")
    exit(1)
    
previous_model = "answerdotai/ModernBERT-base"
print(f"Model used: {previous_model}")
choice = input("[1] Confirm Training: ")

model = AutoModelForSequenceClassification.from_pretrained(previous_model, num_labels=6)
tokenizer = AutoTokenizer.from_pretrained(previous_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class LossCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            
loss_callback = LossCallback()
label_map = {'suicide': 1, 'non-suicide': 0}

if choice == "1":
    df = pd.read_csv('./dataset/clean_suicidal.csv')   

    if 'label' not in df.columns or 'text' not in df.columns:
        print("Error: 'text' or 'label' column is missing from the dataset.")
        exit(1)

    if df['label'].dtype == 'object':
        df['label'] = df['label'].map(label_map)

    if df['label'].isnull().any():
        print("Error: Some labels could not be mapped properly.")
        exit(1)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='label', order=range(len(label_map)), palette='Set2')
    plt.title('Binary Classification Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(ticks=range(len(label_map)), labels=list(label_map.keys()), rotation=45)
    plt.tight_layout()
    plt.show()

    # training split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),  # for debugging : add [:xxxx] x = Integers, after .tolist()
        df['label'].tolist(),   
        test_size=0.2,
        random_state=42
    )
    def tokenize_and_encode(texts, labels):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)  # Add labels as tensor
        }

        if 'token_type_ids' in encodings:
            dataset_dict['token_type_ids'] = encodings['token_type_ids']
        
        return Dataset.from_dict(dataset_dict)
      
    def compute_metrics(p):
        logits, labels = p
        preds = logits.argmax(axis=-1) 

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='binary')   
        precision = precision_score(labels, preds, average='binary', zero_division=1)   
        recall = recall_score(labels, preds, average='binary', zero_division=1)   

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"F1 Score: {f1}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")

        metrics = {
            'eval_accuracy': accuracy,
            'eval_f1': f1,
            'eval_precision': precision,
            'eval_recall': recall
        }
        return metrics

    train_dataset = tokenize_and_encode(train_texts, train_labels)
    val_dataset = tokenize_and_encode(val_texts, val_labels)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, 
                                        early_stopping_threshold=0.01)
    
    outputs_dir = os.path.join(os.getcwd(), 'outputs')
    logs_dir = os.path.join(os.getcwd(), 'logs')

    final_training_args = TrainingArguments(
        output_dir=outputs_dir,
        per_device_train_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.02,
        learning_rate=5e-5,
        warmup_steps=5500,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_strategy="steps",
        logging_dir=logs_dir,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        fp16=True,
        disable_tqdm=False,
        max_grad_norm=2.3 
    )

    trainer = Trainer(
        model=model,
        args=final_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback, early_stopping]
    )

    # trainer.train(checkpoint-xxxx) : if resuming from checkpoint (eg. human error)
    trainer.train()

    saved_title = "stardust_1"
    model.save_pretrained(saved_title)
    tokenizer.save_pretrained(saved_title)
    print("Further fine-tuning completed and model saved.")
    
else:
    print("Invalid choice. Exiting.")
    exit(1)
