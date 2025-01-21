from ollama import chat
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import sounddevice as sd
import numpy as np




# Set seed for reproducibility
seed = 22
torch.manual_seed(seed)
np.random.seed(seed)

# Check if CUDA is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-jenny-30H").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-jenny-30H", add_prefix_space=True)

# Your prompt and description
prompt = "Hey its Jenny.... i'm thinking about Gloria"  # Assuming 'generated_text' contains your desired text for TTS
description = "Jenny's voice is cute and cheerful with English accent."

# Encode description and prompt
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Manually create an attention mask
attention_mask = (input_ids != tokenizer.pad_token_id).long()

# Generate the audio
generation = model.generate(input_ids=input_ids, attention_mask=attention_mask, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

# Play the audio directly
sd.play(audio_arr, samplerate=model.config.sampling_rate)
sd.wait()  # Wait until the audio finishes playing
