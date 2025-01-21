import torch
from datasets import load_dataset, Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import optuna.visualization as vis
import logging
import torch
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaPreTrainedModel,
    EarlyStoppingCallback, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from transformers.modeling_outputs import ModelOutput
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaModel
import torch.nn.functional as F
from dataclasses import dataclass
import torch.quantization
from typing import Optional, Tuple, Union

# Setup logging
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
    
# Specify the new model to use
previous_model = "roberta-base"
print(f"Model used: {previous_model}")
choice = input("[1] Confirm Training: ")


@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class FlashAttentionLayer(nn.Module):
    """Optimized Flash Attention implementation for RoBERTa"""
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} is not divisible by number of attention "
                f"heads {config.num_attention_heads}"
            )
            
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        
        self.dropout = config.attention_probs_dropout_prob
        self.is_causal = False  # RoBERTa doesn't use causal attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.size()

        # Project to QKV
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_length, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Handle attention mask
        if attention_mask is not None:
            # Convert from [batch_size, seq_length] to [batch_size, 1, seq_length, seq_length]
            attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(hidden_states.dtype).min
        
        # Apply Flash Attention
        try:
            from flash_attn import flash_attn_func
            
            # Pack Q, K, V for flash attention
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            
            # Apply Flash Attention
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scaling,
                causal=self.is_causal
            )

            
        except ImportError:
            # Fallback to regular attention if flash_attn is not available
            q = q.transpose(1, 2)  # [batch_size, seq_length, num_heads, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )
            
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2)


        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, -1)
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (None,)  # Flash attention doesn't return attention weights
        
        return outputs




class FlashRobertaLayer(nn.Module):
    """Improved RoBERTa layer with Flash Attention and SwiGLU"""
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = FlashAttentionLayer(config)
        # SwiGLU requires double the intermediate size
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # Fixed: using hidden_size instead of eps
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def swish(self, x):
        """Swish activation function: x * sigmoid(x)"""
        return x * torch.sigmoid(x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        # Pre-LayerNorm architecture for better training stability
        attn_residual = hidden_states
        hidden_states = self.layernorm1(hidden_states)
        
        # We only use the basic attention arguments and ignore the rest
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]
        
        # First residual connection
        hidden_states = attn_residual + self.dropout(attention_output)
        
        # FFN with Pre-LayerNorm
        ffn_residual = hidden_states
        hidden_states = self.layernorm2(hidden_states)
        
        # SwiGLU Feed-Forward
        intermediate_output = self.intermediate(hidden_states)
        gate, transform = intermediate_output.chunk(2, dim=-1)
        layer_output = self.swish(gate) * transform  # SwiGLU formula
        layer_output = self.output(layer_output)
        
        # Second residual connection
        layer_output = ffn_residual + self.dropout(layer_output)
        
        outputs = (layer_output,)
        if output_attentions:
            outputs += attention_outputs[1:]
        
        # Add None for past_key_value if use_cache is enabled
        if use_cache:
            outputs += (None,)
            
        return outputs






class FlashRobertaModel(RobertaModel):
    """RoBERTa model with Flash Attention layers"""
    def __init__(self, config):
        super().__init__(config)
        self.encoder.layer = nn.ModuleList([FlashRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

class FlashRobertaForSequenceClassification(RobertaPreTrainedModel):
    """RoBERTa model with Flash Attention for sequence classification"""
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.roberta = FlashRobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Load models directly



config_6 = RobertaConfig.from_pretrained(previous_model, num_labels=6, 
                                            ignore_mismatched_sizes=True, 
                                            hidden_dropout_prob=0.01,  
                                            attention_probs_dropout_prob=0.01)

model = FlashRobertaForSequenceClassification.from_pretrained(previous_model, config=config_6, strict=False)
tokenizer = RobertaTokenizer.from_pretrained(previous_model)
print(f"Model {previous_model} loaded with Flash Attention")
# Set up the device
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
# Initialize the loss callback
loss_callback = LossCallback()

# Emotion-to-label mapping and reverse mapping

label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
reverse_label_map = {v: k for k, v in label_map.items()} # map vise versa labels


if choice == "1":
    # Load the dataset
    df = pd.read_csv('./dataset/emotions.csv')

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

    # Plot emotion distribution with Seaborn
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label', order=range(len(label_map)), palette='Set2')
    plt.title('Real World Scenario Emotion Distribution')
    plt.xlabel('Emotion Labels')
    plt.ylabel('Frequency')
    plt.xticks(ticks=range(len(label_map)), labels=[label_map[i] for i in range(len(label_map))], rotation=45)
    plt.tight_layout()
    plt.show()
    

    # Split the dataset (reduced size for faster debugging)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),  
        df['label'].tolist(),
        test_size=0.2,
        random_state=42
    )
    

    # Tokenize function that includes labels
    def tokenize_and_encode(texts, labels):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Convert to dictionary and add labels
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)  # Add labels as tensor
        }
        
    # Add token_type_ids only if they exist
        if 'token_type_ids' in encodings:
            dataset_dict['token_type_ids'] = encodings['token_type_ids']
        
        
        return Dataset.from_dict(dataset_dict)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    def compute_metrics(p):
        # Extract logits and true labels
        logits, labels = p
        preds = logits.argmax(axis=-1)  # Convert logits to class predictions

        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average=None)  # F1 score for each class
        precision = precision_score(labels, preds, average='macro', zero_division=1)  # Macro precision
        recall = recall_score(labels, preds, average='macro', zero_division=1)  # Macro recall

        # Log overall metrics
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")

        # Map F1 scores to emotions and log each one
        emotion_f1_scores = {label_map[i]: f1[i] for i in range(len(f1))}
        for emotion, score in emotion_f1_scores.items():
            logger.info(f"F1 score for {emotion}: {score}")

        # Return evaluation metrics
        metrics = {
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
        }

        # Add emotion F1 scores to the dictionary
        metrics.update(emotion_f1_scores)

        return metrics

    # Create datasets with labels
    train_dataset = tokenize_and_encode(train_texts, train_labels)
    val_dataset = tokenize_and_encode(val_texts, val_labels)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, 
                                        early_stopping_threshold=0.01)

    final_training_args = TrainingArguments(
        output_dir='./output',
        per_device_train_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        learning_rate=1e-5,
        warmup_steps=4500,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=100,
        save_steps=1000,
        save_strategy="steps",
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=True,
        disable_tqdm=False,
        max_grad_norm=2.2 
    )
    

    # Initialize trainer with compute_metrics
    trainer = Trainer(
        model=model,
        args=final_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback, early_stopping]
    )

    # Train the model, resuming from last checkpoint *maintain training argusments the same
    trainer.train()

    # Save the fine-tuned model
    saved_title = "stardust_6"
    model.save_pretrained(saved_title)
    tokenizer.save_pretrained(saved_title)
    print("Further fine-tuning completed and model saved.")
    
else:
    print("Invalid choice. Exiting.")
    exit(1)
    