import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import sounddevice as sd
import numpy as np

# set seed (eg. 22-56), maintain consistency for per sequence acoustic modelling
seed = 22
torch.manual_seed(seed)
np.random.seed(seed)
# always check CUDA
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load parlerTTS and tokenizer from HuggingFace
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-jenny-30H").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-jenny-30H", add_prefix_space=True)

# prompt = what to say, description = TTS tone-tuning
prompt = "Hey its Jenny.... i'm thinking about Gloria"  # Assuming 'generated_text' contains your desired text for TTS
description = "Jenny's voice is cute and cheerful with English accent."

# encode prompt and desc to pytorch tensors
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# attention mask
attention_mask = (input_ids != tokenizer.pad_token_id).long()

# audio generation
generation = model.generate(input_ids=input_ids, attention_mask=attention_mask, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

# play audio directly to preferred server, or save audio in .wav format
sd.play(audio_arr, samplerate=model.config.sampling_rate)
sd.wait()  
