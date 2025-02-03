'''

  Framework for update, this code for sentiment analysis task is refactorized from rapid prototyping with minor adjustment and efficiency optimization

'''

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from WeightedCompDictionary import CompositeDictionary
from dataclasses import dataclass

@dataclass
class EmotionPrediction:
    dominant_primary_emotion: str
    dominant_primary_percentage: float
    primary_emotion_probabilities: Dict[str, float]
    dominant_composite_emotion: str
    dominant_composite_percentage: float
    top_n_composite_emotions: Dict[str, float]

class EmotionClassifier:
    def __init__(self, model_path: str, num_labels: int, composite_dictionary: Optional[Dict] = None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,   
            model_max_length=512
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            torchscript=True,   
            return_dict=False   
        )
        
        self.model = self.model.to(self.device).eval()   
        
        self.emotion_to_label = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
        self.composite_dictionary = composite_dictionary or {}

        self.composite_cache = {
            name: (
                torch.tensor(data['emotions'], device=self.device),
                torch.tensor(data['weights'], device=self.device, dtype=torch.float32)
            )
            for name, data in self.composite_dictionary.items()
        }

    @torch.no_grad()  
    def GetEmotionForClassification(
        self,
        texts: str,
        threshold: float = 0.2,
        temperature: float = 1.5,
        max_length: int = 512,
        top_n: int = 3
    ) -> EmotionPrediction:
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
                return_token_type_ids=False,  
                return_attention_mask=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.amp.autocast(device_type='cuda'):
                logits = self.model(**inputs)[0]

                probabilities = torch.softmax(logits / temperature, dim=1)

                probabilities_np = probabilities.cpu().numpy()[0]

            emotion_percentages = {
                self.emotion_to_label[i]: float(prob * 100)
                for i, prob in enumerate(probabilities_np)
            }
            total_prob = sum(emotion_percentages.values())
            emotion_percentages = {
                k: v / total_prob * 100 
                for k, v in emotion_percentages.items()
            }
            filtered_emotions = {
                emotion: prob 
                for emotion, prob in emotion_percentages.items() 
                if prob >= threshold
            }
            dominant_emotion, max_prob = max(
                filtered_emotions.items(),
                key=lambda x: x[1],
                default=("neutral", 0)
            )
            composite_probs = {}
            for name, (indices, weights) in self.composite_cache.items():
                selected_probs = probabilities.index_select(1, indices)
                score = (selected_probs * weights).sum().item()
                composite_probs[name] = score * 100
            filtered_composites = {
                k: v for k, v in composite_probs.items() 
                if v >= threshold
            }
            sorted_composites = dict(
                sorted(
                    filtered_composites.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
            )
            dominant_composite = max(
                sorted_composites.items(),
                key=lambda x: x[1],
                default=("neutral", 0)
            )
            return EmotionPrediction(
                dominant_primary_emotion=dominant_emotion,
                dominant_primary_percentage=max_prob,
                primary_emotion_probabilities=emotion_percentages,
                dominant_composite_emotion=dominant_composite[0],
                dominant_composite_percentage=dominant_composite[1],
                top_n_composite_emotions=sorted_composites
            )
        except Exception as e:
            print(f"Error in emotion classification: {str(e)}")
            return EmotionPrediction(
                dominant_primary_emotion="Error",
                dominant_primary_percentage=0.0,
                primary_emotion_probabilities={},
                dominant_composite_emotion="Error",
                dominant_composite_percentage=0.0,
                top_n_composite_emotions={}
            )
'''

  Example class usage

'''


if __name__ == "__main__":
    local_directory = os.getcwd()
    classifier = EmotionClassifier(
        model_path=f"{local_directory}/CoreDynamics/models/stardust_6",
        num_labels=6,
        composite_dictionary=CompositeDictionary
    )
    
    result = classifier.GetEmotionForClassification(
        "i feel so happy today",
        threshold=0.2,
        temperature=2.9,
        max_length=512,
        top_n=5
    )
    print(result.primary_emotion_probabilities)
    print(result.top_n_composite_emotions)
