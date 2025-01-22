import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
import os
# uncomment below if using wayland compositor
# os.environ['QT_QPA_PLATFORM'] = 'xcb'
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA current device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print(f"CUDA is not available: {torch.cuda.is_available()}")
    exit(1)
    
# specify pretrained model (eg. BERT-large, RoBERTa-base)
# for now, ModernBERT excels significantly than previous BERT models
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
# loss callback : computer metrics
loss_callback = LossCallback()
# reverse map dataset > trainer
label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
reverse_label_map = {v: k for k, v in label_map.items()} # map vise versa labels

if choice == "1":
    # dataset load
    df = pd.read_csv('./dataset/emotions.csv')
    # check 'label' column, if doesn't exist, manually overwrite with 'label'
    if 'label' not in df.columns:
        print("Error: 'label' column is missing from the dataset.")
        exit(1)
    # text labels to numeric if they're not already numeric
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map(reverse_label_map)
    # validate conversion
    if df['label'].isnull().any():
        print("Error: Some labels could not be mapped properly.")
        exit(1)
    # plot dataset distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label', order=range(len(label_map)), palette='Set2')
    plt.title('Real World Scenario Emotion Distribution')
    plt.xlabel('Emotion Labels')
    plt.ylabel('Frequency')
    plt.xticks(ticks=range(len(label_map)), labels=[label_map[i] for i in range(len(label_map))], rotation=45)
    plt.tight_layout()
    plt.show()
    # training split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),  # for debugging purpose : add [:xxxx] after .tolist()
        df['label'].tolist(),
        test_size=0.2,
        random_state=42
    )
    # tokenize labels
    def tokenize_and_encode(texts, labels):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        # dictionary conversion to labels
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)  # specify labels as tensors 
        }
        
    # add token_type_ids only if they exist
        if 'token_type_ids' in encodings:
            dataset_dict['token_type_ids'] = encodings['token_type_ids']
        
        return Dataset.from_dict(dataset_dict) 
        
    def compute_metrics(p):
        # extract logits and true labels
        logits, labels = p
        # convert logits to class predictions
        preds = logits.argmax(axis=-1) 
        # calculate metrics
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average=None)  # F1 score for each class
        precision = precision_score(labels, preds, average='macro', zero_division=1)  # Macro precision
        recall = recall_score(labels, preds, average='macro', zero_division=1)  # Macro recall
        # log overall metrics
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        # map F1 scores to emotions and log each one
        emotion_f1_scores = {label_map[i]: f1[i] for i in range(len(f1))}
        for emotion, score in emotion_f1_scores.items():
            logger.info(f"F1 score for {emotion}: {score}")
        # return evaluation metrics
        metrics = {
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
        }
        # add each features/emotion F1 scores to the dictionary
        metrics.update(emotion_f1_scores)

        return metrics

    # create datasets with labels
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
        weight_decay=0.01,
        learning_rate=1e-6,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_strategy="steps",
        logging_dir=logs_dir,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # report_to="none",
        fp16=True,
        disable_tqdm=False,
        max_grad_norm=2.2 
    )
    # initialize trainer with compute_metrics
    trainer = Trainer(
        model=model,
        args=final_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback, early_stopping]
    )
    # trainer.train(checkpoint-xxxx) if resume training (eg. power loss, short circuits, human error)
    trainer.train()
    # save model
    saved_title = "stardust_6"
    model.save_pretrained(saved_title)
    tokenizer.save_pretrained(saved_title)
    print("Further fine-tuning completed and model saved.")
else:
    print("Invalid choice. Exiting.")
    exit(1)
    
