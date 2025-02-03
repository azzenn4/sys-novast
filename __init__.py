'''
  dummy init because i lose myself

  the writer of this script feels like its duct-taping a llama to spaceX starship :
  
  the main script for running sys-novast, the code is somehow cluttered and needs to introduce code modularity for .py files with specialized function to-tasks

'''




import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import numpy as np
import pytesseract
from mss import mss
import cv2
import os
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from pynput.mouse import Controller
from sklearn.preprocessing import StandardScaler
import multiprocessing
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
import re
from ollama import chat 
from parler_tts import ParlerTTSForConditionalGeneration
import threading
import queue

os.environ["TRANSFORMER_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA current device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print(f"CUDA is not available: {torch.cuda.is_available()}")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# example model trained using ./trainer/{mutliclass trainer}.py
# model : multiclass classification -> 6 labels
tokenizer_6 = AutoTokenizer.from_pretrained("./models/stardust_6")
model_6 = AutoModelForSequenceClassification.from_pretrained("./models/stardust_6", num_labels=6)
model_6 = model_6.to(device)

# example model trained using ./trainer/{binary trainer}.py
# model : binary classification -> 2 labels
tokenizer_1 = AutoTokenizer.from_pretrained("./models/stardust_1")
model_ax1 = AutoModelForSequenceClassification.from_pretrained("./models/stardust_1", num_labels=2)
model_ax1 = model_ax1.to(device)

if torch.cuda.is_available():
    print("Final GPU Memory Allocated:", torch.cuda.memory_allocated())
    print("Final GPU Memory Reserved:", torch.cuda.memory_reserved())
    
print(model_ax1)

'''

    Composite Emotions, inspired by :
    
        Paul Ekman's Theory Of Emotions
        Russels J. A Dimension of Emotion
        Richard Lazarus Cognitive Appraisal
        Jaak Panksepp Affective Neuroscience
    
'''



'''

    Logits Post Processing for Multiclass

'''
'''

    Multiclass Logits Post Processing are highly optional for generic tasks,

    Since humans process emotions differently and biologically for each individual,
    its safe to say that there are no fixed rule for such a thing

    .Thus,
    
    it all comes down to user personal preference to determine NovaAstra characteristics
    for processing emotion extraction. Though this is highly technical and may require
    holistic approach.

    For example,
    without Logits Postprocessing :

        Joy : 99.8%
        Anger : 0.2%
        Fear : 0.0%
        Surprise : 0.0%
        Love : 0.0%
        Sadness : 0.0%

    Summarizing the example, we know that NovaAstra will perceive emotion
    more vividly. But this comes at a downside, which is the lack of diverse
    emotional-point-of-view and lack of contextual composite emotion-awareness.

    The solution is to implement temperature scaling with thresholds,
    simulating a more comprehensive emotional intelligence found in
    thoughtful individual.
    
'''

def apply_temperature_scaling(logits, temperature):
    return logits / temperature

def apply_confidence_threshold(probabilities, threshold):
    return {emotion: prob for emotion, prob in probabilities.items() if prob >= threshold}

emotion_to_label = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
label_map = {v: k for k, v in emotion_to_label.items()} 

def get_emotion(texts, threshold=0.2, temperature=1.0, top_n=3):
    try:
        inputs = tokenizer_6(texts, return_tensors="pt", truncation=True, padding=True, max_length=128*2).to(device)
        # Disable gradient accumulation, since its an inference not training
        with torch.no_grad(): 
            # GPU must support FP16 computation.. , by using amp autocast on GPU, theres a minimal but noticeable inference speed
            with torch.amp.autocast(device_type='cuda', enabled=True):  
                outputs = model_6(**inputs)
                logits = outputs.logits
        logits = apply_temperature_scaling(logits, temperature)
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        emotion_percentages = {emotion_to_label[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}
        total_prob = sum(emotion_percentages.values())
        emotion_percentages = {emotion: float((prob / total_prob) * 100) for emotion, prob in emotion_percentages.items()}
        filtered_emotions = apply_confidence_threshold(emotion_percentages, threshold)
        # fallback option = neutral, though highly-unlikely with post-processing
        dominant_emotion = max(filtered_emotions.items(), key=lambda x: x[1], default=("neutral", 0))[0] 
        composite_emotion_probabilities = {}
        for composite_emotion, emotion_indices in composite_emotions.items():
            composite_probs = [probabilities[i] for i in emotion_indices]
            weighted_sum = sum(composite_probs)  
            composite_emotion_probabilities[composite_emotion] = float(round(weighted_sum / len(composite_probs) * 100, 2))
        filtered_composite_emotions = apply_confidence_threshold(composite_emotion_probabilities, threshold)
        sorted_composites = sorted(filtered_composite_emotions.items(), key=lambda x: x[1], reverse=True)
        top_composites = dict(sorted_composites[:top_n])
        dominant_composite_emotion = max(top_composites.items(), key=lambda x: x[1], default=("neutral", 0))[0]
        return {
            "dominant_primary_emotion": dominant_emotion,
            "dominant_primary_percentage": emotion_percentages.get(dominant_emotion, 0),
            "primary_emotion_probabilities": emotion_percentages,
            "dominant_composite_emotion": dominant_composite_emotion,
            "dominant_composite_percentage": top_composites.get(dominant_composite_emotion, 0),
            "top_composite_emotions": top_composites,
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "dominant_primary_emotion": "Error",
            "dominant_primary_percentage": 0,
            "primary_emotion_percentages": {},
            "dominant_composite_emotion": "Error",
            "dominant_composite_percentage": 0,
            "top_composite_emotions": {},
        }


'''

    Logits Post Processing for Binary

'''

'''

 For now. Currently, the models are fine tuned off "almost relevant" datasets to simulate empathy
 by suicide and non-suicide probabilities, in the meantime it is expected
 to fine-tune the model with a more relevant dataset. 
 This is due to initial goal for NovaAstra as a Psychologist that handle
 tests for suicidal subjects in a University Project. 

 But please be cautious when using the quantized models for Mental-Health related task, this is due to
 a significantly lower accuracy for analysis tasks that applies to both Binary and Multiclass.

'''
suicide_label = {'suicide': 1, 'non-suicide': 0}
def get_suicide_risk(texts):
    try:
        inputs = tokenizer_1(texts, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=True):  
                outputs = model_ax1(**inputs)
                logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        suicide_prob = float(probabilities[1])  # Convert to float
        non_suicide_prob = float(probabilities[0])  # Convert to float
        predicted_label = 'suicide' if suicide_prob > non_suicide_prob else 'non-suicide'
        return {
            'label': predicted_label,
            'probabilities': {
                'label': predicted_label,
                'suicide_prob': suicide_prob,  # return as float
                'non_suicide_prob': non_suicide_prob  # return as float
            }
        }
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

'''

    VECTOR DATABASE (Conversation Memory)

'''
dimension = 768  # RoBERTa base model has 768-dimensional embeddings
index = faiss.IndexFlatL2(dimension)
metadata = []

def get_embedding(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        if isinstance(model, AutoModelForSequenceClassification):
            # for RoBERTa
            outputs = model.roberta(**inputs.to(device), output_hidden_states=True)
        else:
            # for models like RoBERTa or others where embeddings are needed
            outputs = model(**inputs.to(device), output_hidden_states=True)
    hidden_states = outputs.hidden_states
    embedding = hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
    
    return embedding

def log_results(texts, model, tokenizer_6, device):
    global metadata
    embedding_6 = get_embedding(texts, model, tokenizer_6, device)
    # Check if the embedding already exists in the FAISS index (to avoid duplicates)
    D, I = index.search(np.array([embedding_6]), k=1)  # Search for the nearest neighbor
    # If the distance is below a threshold, it indicates that the text is similar to an existing one
    # You can adjust the threshold based on your dataset
    if D[0][0] < 0.8:  # Example threshold, adjust as necessary
        return  # Skip storage if it's a duplicate
    emotion_result = get_emotion(texts)
    suicide_result = get_suicide_risk(texts)
    primary_emotion_probabilities = emotion_result["primary_emotion_probabilities"]
    # print(primary_emotion_probabilities) # uncomment for debugging
    log_data = {
        "text": texts,
        "primary_emotion_probabilities": primary_emotion_probabilities,
        "suicide_risk_probabilities": suicide_result["probabilities"] 
    }
    # add the embedding to the FAISS index 
    index.add(np.array([embedding_6]))
    # append the log data to metadata
    metadata.append(log_data)
    
# normalize embeddings (for cosine similarity)
def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
# cosine similarity between two vectors
def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

'''
dim = 1
[  [[2.3, 1.2, 0.5, 0.3, 1.0, 0.8],  # Forecast Step 1 (Batch 1)
   [1.5, 2.2, 0.7, 0.8, 0.9, 1.4],  # Forecast Step 2 (Batch 1)
   [2.1, 1.7, 0.6, 1.1, 0.4, 0.9]], # Forecast Step 3 (Batch 1)

  [[1.0, 1.5, 0.4, 2.2, 0.7, 1.3],  # Forecast Step 1 (Batch 2)
   [0.9, 2.0, 0.8, 1.0, 1.4, 0.5],  # Forecast Step 2 (Batch 2)
   [1.2, 1.3, 1.1, 0.7, 1.6, 1.0]]  # Forecast Step 3 (Batch 2)
]
dim = 2 
[
  [[0.32, 0.16, 0.09, 0.07, 0.18, 0.14],  # Probabilities for Step 1 (Batch 1)
   [0.21, 0.35, 0.12, 0.13, 0.15, 0.24],  # Probabilities for Step 2 (Batch 1)
   [0.29, 0.25, 0.11, 0.18, 0.07, 0.10]], # Probabilities for Step 3 (Batch 1)

  [[0.12, 0.19, 0.07, 0.38, 0.09, 0.15],  # Probabilities for Step 1 (Batch 2)
   [0.10, 0.30, 0.14, 0.15, 0.21, 0.10],  # Probabilities for Step 2 (Batch 2)
   [0.15, 0.17, 0.15, 0.10, 0.23, 0.20]]  # Probabilities for Step 3 (Batch 2)

   Planning for bidirectional LSTM for better prediction, currently : unidirectional;
]
'''

'''

    Unidirectional LSTM,

    The unidirectional LSTM is used to simulate how an individual perceives emotions during a conversation. 
    Instead of evaluating emotions after each conversational turn (e.g., sentence by sentence), 
    the LSTM processes the entire sequence of metadata, calculating the average emotion across the sequence length.
    
    This approach reflects how humans often judge emotions holistically, after accumulating enough context, 
    rather than immediately interpreting emotions from isolated sentences or short exchanges. 
    By considering the entire conversation history, the model aims to make a more informed and contextually aware emotional judgment.    
    The LSTM is chosen for its ability to handle sequential data effectively, 
    ensuring the final output represents a well-rounded emotional context based on the conversation's overall flow.
    
'''
def prepare_emotion_sequences(metadata, seq_length, forecast_steps):
    # extract primary emotion probabilities
    emotion_probabilities = []
    for entry in metadata:
        if "primary_emotion_probabilities" in entry:
            primary_emotions = entry["primary_emotion_probabilities"]
            
            if isinstance(primary_emotions, dict):
                # Convert percentages to probabilities (divide by 100)
                probabilities = [value / 100.0 for value in primary_emotions.values()]  # Convert percentages to probabilities
                # Print the original percentages and the converted probabilities
                # print("Original percentages:", primary_emotions) # uncomment for debugging
                # print("Converted probabilities:", probabilities) # uncomment for debugging
                emotion_probabilities.append(probabilities)
            else:
                print(f"Error: Expected 'primary_emotion_probabilities' to be a dictionary in entry {entry}")
        else:
            print(f"Error: Missing 'primary_emotion_probabilities' in entry {entry}")
    emotion_probabilities = np.array(emotion_probabilities, dtype=float)
    if emotion_probabilities.shape[0] == 0:
        raise ValueError("No valid emotion probabilities found in metadata.")
    if len(emotion_probabilities) < (seq_length + forecast_steps):
        print(f"Error: Not enough data points to create sequences. Required: {seq_length + forecast_steps}, Found: {len(emotion_probabilities)}")
        return np.array([]), np.array([])
    # Get the number of features (emotions)
    num_features = emotion_probabilities.shape[1]
    # Create sequences for multi-step forecasting
    X, y = prepare_sequences(emotion_probabilities, seq_length, forecast_steps)
    return X, y, num_features

def prepare_sequences(data, seq_length, forecast_steps):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_steps):
        X.append(data[i:i+seq_length])
        # Forecast the next `forecast_steps` points
        y.append(data[i+seq_length:i+seq_length+forecast_steps])  
    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast_steps, num_features):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps
        self.num_features = num_features        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # output size = num_features * forecast_steps
        self.fc = nn.Linear(hidden_size, num_features * forecast_steps)  

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        out, _ = self.lstm(x, (h0, c0))
         # output for the last time step in the sequence
        out = self.fc(out[:, -1, :]) 
        # reshape output > (batch_size, forecast_steps, num_features)
        out = out.view(-1, self.forecast_steps, self.num_features)  
        # softmax across features (emotions)
        out = F.softmax(out, dim=2)  
        return out

# normalize data for training. This is optional, but often helps LSTMs perform better
def normalize_data(X, y):
    # check the structure and type of X
    # print(f"X type: {type(X)}") # uncomment for debugging
    # print(f"X shape: {X.shape}") # uncomment for debugging
    # inspect the first few rows of X
    # print("First 5 rows of X:", X[:5]) # uncomment for debugging
    # check for NaN or Inf values in X
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: X contains NaN or Inf values.")
        # replace NaN/Inf with zero (or another suitable value)
        X = np.nan_to_num(X)
    # check if X is in the correct numeric format (float)
    if X.dtype != np.float32 and X.dtype != np.float64:
        print("Converting X to float type...")
        # convert all values to float
        X = X.astype(float)  

    # ff X contains a list of dictionaries, convert it to a numerical array
    if isinstance(X, list) and isinstance(X[0], dict):
        print("Converting dictionaries to lists of values...")
        # convert dict to list of values
        X = np.array([list(entry.values()) for entry in X])  
        # debugging line to check the shape
        print(f"X after conversion: {X.shape}")  
    
    # reshape X to 2D for scaling (samples * timesteps, features)
    # flatten sequence dimension (52 * 5, 6)
    X_reshaped = X.reshape(-1, X.shape[-1])  
    
    # initialize to fit the scaler
    scaler = StandardScaler()
     # apply scaling
    X_scaled_reshaped = scaler.fit_transform(X_reshaped) 
    
    # reshape back to the original 3D shape
    # (52, 5, 6)
    X_scaled = X_scaled_reshaped.reshape(X.shape[0], X.shape[1], X.shape[2])  
    
    # check y is a numpy array and scale it if necessary
    # convert y to float if it's not already
    y_scaled = np.array(y, dtype=float)  
    
    # check the scaled data
    # print(f"X_scaled shape: {X_scaled.shape}") # uncomment for debugging
    # print(f"y_scaled shape: {y_scaled.shape}") # uncomment for debugging
    
    return X_scaled, y_scaled, scaler


def train_lstm(X_tensor, y_tensor, input_size, hidden_size, num_layers, num_epochs, learning_rate, forecast_steps, num_features):
    # initialize model && parameter
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        forecast_steps=forecast_steps,
        num_features=num_features
    )
    # set criterion - crossentropyloss, since we have 6 label..
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()       
        '''
        
        # for debugging
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
        '''
    return model

def forecast_lstm(model, X_tensor, scaler, forecast_steps, num_features):
    model.eval()
    # take last sequence
    last_sequence = X_tensor[-1].unsqueeze(0)  
    with torch.no_grad():
        forecast = model(last_sequence)
    forecast = forecast.squeeze().numpy()
    # reshape forecast to (forecast_steps, num_features)
    forecast = forecast.reshape(forecast_steps, num_features)
    # apply inverse_transform to each time step
    forecast_original_scale = np.zeros_like(forecast)
    for i in range(forecast_steps):
        # reshape each time step to (1, num_features) for inverse transform
        forecast_original_scale[i] = scaler.inverse_transform(forecast[i].reshape(1, -1))
    
    return forecast_original_scale

def train_and_forecast(metadata, seq_length, forecast_steps, hidden_size, num_layers, num_epochs, learning_rate):
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("Preparing sequences...")
        X, y, num_features = prepare_emotion_sequences(metadata, seq_length, forecast_steps)
        
        if X.size == 0 or y.size == 0:
            return {
                "forecast": [],
                "error": "Prepared sequences are empty. Check input data."
            }

        print("Normalizing data...")
        X_scaled, y_scaled, scaler = normalize_data(X, y)
        
        if X_scaled.size == 0 or y_scaled.size == 0:
            return {
                "forecast": [],
                "error": "Normalized data is empty. Check data processing steps."
            }
        
        print("Converting to tensors...")
        # change to float 16 for better perfomance // cost of accuracy
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32) 
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32) 
        
        print("Training LSTM model...")
        model = train_lstm(
            X_tensor=X_tensor,
            y_tensor=y_tensor,
            input_size=X_tensor.shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            forecast_steps=forecast_steps,
            num_features=num_features
        )
        
        print("Making forecasts...")
        forecast = forecast_lstm(model, X_tensor, scaler, forecast_steps, num_features)

        # return the forecast in the expected dictionary format
        return {
            # convert numpy array to list for better compatibility
            "forecast": forecast.tolist(),  
            "error": None
        }
    
    except Exception as e:
        return {
            "forecast": [],
            "error": str(e)
        }

def predict_emotions(metadata, seq_length, forecast_steps, hidden_size, num_layers, 
                    num_epochs, learning_rate):
    result = train_and_forecast(metadata, seq_length, forecast_steps, hidden_size, 
                              num_layers, num_epochs, learning_rate)
    if result.get("error") is not None:
        print(f"Error: {result['error']}")
        return
    forecast = result['forecast']
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return forecast  
'''

    Text Generation : Llama3.2 3B (For most hardware (RTX with VRAM > 4GB))
                      Llama3.3 70B (For workstations (RTX / A series with VRAM > 24GB))

    Make sure that you have Ollama installed locally with Llama3.2 Model.

    For documentation please refer to https://github.com/ollama/ollama

'''
def text_gen_only(tcp):
    global cast_iter
    global metadata
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    def get_gpu_memory_info():
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(current_device).total_memory / 1024**2
            allocated = torch.cuda.memory_allocated(current_device) / 1024**2
            free = total - allocated
            return {'total': total, 'allocated': allocated, 'free': free}
        return None
    # set threshold to your hardware capability
    def get_device(threshold_vram_mb=200):  
        if torch.cuda.is_available():
            memory_info = get_gpu_memory_info()
            if memory_info and memory_info['free'] >= threshold_vram_mb:
                return "cuda:0"
        return "cpu"

    try:
        subject_name = "My friend"
        last_ = metadata[-1]
        _texts = [entry["text"] for entry in metadata[-10:] if "text" in entry]  
        primary_ = last_['primary_emotion_probabilities']
        primary_emo = max(primary_, key=primary_.get)
        empathy_ = last_['suicide_risk_probabilities']
        empathy_rate = empathy_['suicide_prob'] * 100
        conv_ = emo_flow[-1]
        conv_flow = conv_[0]
        # what emotion to react as relevant to opposite emotion (NovaAstra | Subject)
        emotion_to_reaction = {
            'sadness': "compassion", 'joy': "cheerfulness",
            'love': "affection", 'anger': "calmness",
            'fear': "reassurance", 'surprise': "excitement"
        }
        react_ = emotion_to_reaction.get(conv_flow, "")

        print(f"Subject : {subject_name}")
        print(f"primary emotion : {primary_emo} || empathy rate : {empathy_rate} || conv flow : {conv_flow}")
        print(f"TEXT : {_texts}")
        # currently hard-coded prompts, in the meantime it should use generated reasoning like deepseek-r1, however this affects performance significantly
        # structure : future : 
      
      # prompt/reasoning : prompt > fine tuning | reasoning extract only <think> tags from deepseek r1
      
    
        prompt_parts = [
            f"You're currently talking with {subject_name}, he/she is feeling {primary_emo} right now.", # < subject introduction
            f"He/she is saying this : {_texts} lately." # < subject context
            f"it seems that {subject_name} is experiencing {next(iter(tcp))} right now, however you can feel that the conversation is going to be full of {conv_flow}", # < simulate reaction 
            f"React something about {subject_name} condition,  Give your advice about {subject_name} condition to me", # <  TODO
            f"Respond with {react_} on giving advice about {subject_name} statement and emotions its feeling,", # < TODO REACT
            f"dont ask any question just provide your genuine reaction about what {subject_name} just said." # TODO CONSTRAINT
        ]
        rest_ = " ".join(prompt_parts)
        del prompt_parts  #   free memory
        print("Thinking...")
        stream = chat(
            model='llama3.2',
            # changed role to 'assistant' from 'user'
            messages=[{'role': 'assistant', 'content': rest_}],
            stream=False, # set to false, currently opencv dev interface doesn't support sequential streaming 
            options={
                "num_ctx": 8192,  
                "f16_kv": True, #  set to FP16
                "seed": 42,
                "num_batch": 16 # for consumer grade, (eg., 30, 40 series) set 16-32
            }
        )
        generated_text = "".join(chunk['message']['content'] for chunk in stream)
        # print(f"Llama said : {generated_text}") # uncomment for debugging
        global jenny_color 
        jenny_color = (244, 0, 252)
        # empty metadata
        metadata = []
        # print(f"Metadata is cleared after Jenny's speech. Proof :{metadata}") # uncomment for debugging
    except Exception as e:
        print(f"Error in voice generation: {e}")
        return None
    finally:
       # cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        generated_text = "Jenny :" + generated_text
        # text generation is done via separate thread for efficiency :
        gen_queue.put(generated_text)
'''

    Since OpenCV doesn't support word wrapping :

'''
def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = ''
    for word in words:
        test_line = current_line + (' ' if current_line else '') + word
        (w, h), _ = cv2.getTextSize(test_line, font[0], font[1], font[2])
        if w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


jenny_color = (244, 0, 252)
emo_flow = []
# preprocess function for OCR
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh
# function to clean text from OCR output
def clean_text(text):
    cleaned_text = " ".join(text.split())
    filtered_texts = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_text)  # Remove non-alphanumeric characters
    return filtered_texts
# main ocr processing
def ocr_worker(thresh):
    try:
        # PSM 3 for simpler block-based text detection, adjust as needed
        custom_config = r'--oem 3 --psm 3'  
        raw_text = pytesseract.image_to_string(thresh, config=custom_config)
        cleaned_text = clean_text(raw_text)
        # send the text via multiprocessing pipe (eg. process(1) -> process(2))
        send_text.send(cleaned_text) # >>
    except Exception as e:
        print(f"Error in OCR thread: {str(e)}")
        
'''

NovaAstra Interface - OpenCV

We are currently planning to use Qt6; 
however, we have noticed that Qt6 consumes a significant amount of VRAM and RAM, 
and also unnecessary since the end goal of sys-nova is running headless. 
Therefore, 
we are considering that the OpenCV interface may only be necessary for early stages.

We hope that those who have contributed to this project can better understand the context and potentially provide suggestions or solutions.
    
'''
def capture_and_detect():
    sct = mss()
    monitors = sct.monitors
    # get primary monitor (monitor 1), set monitors[0] if you have single monitor
    monitor_ = monitors[1]   
    SCREEN_WIDTH = monitor_['width']
    SCREEN_HEIGHT = monitor_['height']
    frames = 0
    max_visible_lines = 10
    trend_summary = []
    cast_iter = 0
    mouse = Controller()
    global jenny_color
    font = cv2.FONT_HERSHEY_COMPLEX
    met_len = 0
    # initial text
    generated_text = "Jenny haven't think about something..."
    with ProcessPoolExecutor(max_workers=8) as executor:
        while True:
            try:
                global metadata
                global emo_flow
                met_len = len(metadata)
                if frames < 50:
                    region = {'top': 540, 'left': 540, 'width': 300, 'height': 150}

                mouse_x, mouse_y = mouse.position
                monitor_left = monitor_['left']
                monitor_top = monitor_['top']
                relative_mouse_x = mouse_x - monitor_left
                relative_mouse_y = mouse_y - monitor_top
                region['left'] = relative_mouse_x - (region['width'] // 2)
                region['top'] = relative_mouse_y - (region['height'] // 2)
                region['top'] = max(0, min(region['top'], SCREEN_HEIGHT - region['height']))
                region['left'] = max(0, min(region['left'], SCREEN_WIDTH - region['width']))
                capture_region = {
                    'top': region['top'] + monitor_top,
                    'left': region['left'] + monitor_left,
                    'width': region['width'],
                    'height': region['height']
                }
        
                full_screen_frame = np.array(sct.grab(monitor_))
                region_frame = np.array(sct.grab(capture_region))
                region_frame = cv2.cvtColor(region_frame, cv2.COLOR_BGRA2BGR)
                future_thresh = executor.submit(preprocess_frame, region_frame)
                thresh = future_thresh.result()
                executor.submit(ocr_worker, thresh)
                detected_texts = ""
                pep = []
                tcp = []
                srsk = 0.0
                if receive_text.poll():
                    try:
                        detected_texts = receive_text.recv() # <<
                        cv2.putText(full_screen_frame, f"Preview : {detected_texts}", (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 370), font, 1.1, (255, 255, 255), 1, cv2.LINE_AA)
                        emotion_data = get_emotion(detected_texts)
                        suicide_risk_data = get_suicide_risk(detected_texts)
                        pep = emotion_data["primary_emotion_probabilities"]
                        tcp = emotion_data["top_composite_emotions"]
                        srsk = suicide_risk_data["probabilities"]["suicide_prob"]

                        if frames % 10 == 0:
                            log_results(detected_texts, model_6, tokenizer_6, device)
                            # generate response every 70 frames
                            if met_len % 70 == 0 and met_len != 0:
                                generated_text = "Jenny about to say something......"
                                jenny_color = (24, 3, 255)

                                forecast = predict_emotions(metadata, 
                                                            seq_length=65,          # moderate sequence length
                                                            forecast_steps=5,       # predict 5 steps forward
                                                            hidden_size=8,         
                                                            num_layers=1,           # use 2 LSTM layers to avoid overfitting
                                                            num_epochs=5,          # reduce number of epochs to prevent overfitting
                                                            learning_rate=1e-2)     # lower learning rate for stable training
                                # update Summary 
                                trend_summary = []
                                # calculate average probabilities across all time steps
                                avg_probs = {emotion: sum(step[i] for step in forecast) / len(forecast) 
                                            for i, emotion in emotion_to_label.items()}
                                # sort and display average emotions
                                sorted_avgs = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
                                trend_summary.append("Predicted Conversational Flow:")
                                total_avg = sum(prob for _, prob in sorted_avgs)
                                
                                for emotion, avg in sorted_avgs:
                                    percentage = (avg / total_avg * 100) if total_avg > 0 else 0
                                    trend_summary.append(f"{emotion.capitalize():8}: {percentage:.1f}% probability in the next '20' turns")
                                emo_flow = sorted(sorted_avgs, key=lambda x: x[1], reverse=False)                                     
                                trend_summary.append(f"Renewal of metadata || Iteration :: [ {cast_iter} ] :: ")
                                # empty GPU memory for text generation
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                threading.Thread(target=text_gen_only, args=(tcp,)).start()
                                cast_iter += 1
                    except Exception as e:
                        print(f"Error processing text: {str(e)}")
                else:
                    pass
                # display base information
                text_x = 10
                base_text_y = 50
                cv2.putText(full_screen_frame, f"PEP: {pep}", (text_x, base_text_y), font, 1.1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(full_screen_frame, f"TCE: {tcp}", (text_x, base_text_y + 40), font, 1.1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(full_screen_frame, f"EMPATHY: {srsk * 100:.2f}%", (text_x, base_text_y + 80), font, 1.1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(full_screen_frame, f"Frames: {frames}", (text_x, base_text_y + 130), font, 1.1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(full_screen_frame, f"Metadata: {met_len}", (text_x, base_text_y + 170), font, 1.1, (255, 255, 255), 1, cv2.LINE_AA)
                vram_bytes = torch.cuda.memory_allocated(device='cuda')
                vram_gb = vram_bytes / (1024 ** 3)  # convert bytes to gigabytes
                cv2.putText(full_screen_frame, f"Allocation: {vram_gb:.2f} Gigabytes", (text_x, base_text_y + 210), font, 1.1, (255, 255, 255), 1, cv2.LINE_AA)
                                                                                                       
                if not gen_queue.empty():
                    generated_text = gen_queue.get()    
                # 100 margin between screen edge
                max_width = full_screen_frame.shape[1] - 100 

                lines = wrap_text(generated_text, (font, 0.9, 1), max_width)
                for i, line in enumerate(lines):
                    # update y position for each line
                    text_y = (base_text_y + 630) + i * 30  
                    cv2.putText(full_screen_frame, f"{line}", (text_x + 20, text_y), font, 0.9, jenny_color, 1, cv2.LINE_AA)
                # set the initial Y position for the scrolling text
                emotion_text_y = 100
                line_height = 25
                # calculate the starting Y position for trend summary
                trend_start_y = emotion_text_y + (max_visible_lines * line_height) + 30
                # display trend summary with safety checks
                # only process if trend_summary is not empty
                if trend_summary:  
                    trend_summary_len = len(trend_summary)
                    # additional check for safety
                    if trend_summary_len > 0:  
                        for j, line in enumerate(trend_summary):
                            # ensure to not exceed the list bounds
                            trend_line_index = j % trend_summary_len
                            cv2.putText(full_screen_frame, trend_summary[trend_line_index],
                                    (10 + 10, trend_start_y + (j * line_height)),
                                    font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                # draw blue rectangle around detection region
                x_offset = region['left']
                y_offset = region['top']
                cv2.rectangle(full_screen_frame, (x_offset, y_offset), 
                            (x_offset + region['width'], y_offset + region['height']), 
                            (255, 0, 0), 2) 
                # resize frame for smaller window
                scaled_frame = cv2.resize(full_screen_frame, (1200, 600))
                frames += 1
                # display the frame
                cv2.imshow("Jenny", scaled_frame)
                # empty GPU cache every frames
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            except Exception as e:
                print(f"Error : {e} . Likely needs more information to run")
                pass
if __name__ == "__main__":
    # unidirectional pipe, one way set 'duplex=False', two way set 'duplex=True' : which is not in this case, and shouldn't be
    receive_text, send_text = multiprocessing.Pipe(duplex=False)  
    # queue for LLAMA3.2 (since it runs on separate thread to maintain interface responsiveness
    gen_queue = queue.Queue()
    # capture_and_detect is the main process..
    capture_and_detect()
    
    





        


    
