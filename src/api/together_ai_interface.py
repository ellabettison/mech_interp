#!/usr/bin/env python3
"""
Together AI interface for cross-model transfer testing.
"""

import os
import time
import logging
from typing import Optional
import requests
from model_calling.LLM import LLM

logger = logging.getLogger(__name__)

class TogetherAIInterface(LLM):
    """Interface for Together AI models."""
    
    def __init__(self, api_key: str, model: str = "togethercomputer/llama-2-7b-chat"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.temperature = 0.0
    
    def call_model(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1000) -> str:
        """Call Together AI model with a prompt."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use chat completions format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.1
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"Unexpected response format: {result}")
                return ""
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"Together AI API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error details: {error_detail}")
                except:
                    logger.error(f"Response text: {e.response.text}")
            return ""
        except Exception as e:
            logger.error(f"Together AI API error: {e}")
            return ""

# Available Together AI models (serverless)
TOGETHER_MODELS = {
    "llama-3-8b": "meta-llama/Llama-3-8b-chat-hf",
    "llama-3-70b": "meta-llama/Llama-3-70b-chat-hf",
    "llama-2-70b": "meta-llama/Llama-2-70b-hf",
    "gemma-2-27b": "google/gemma-2-27b-it",
    "gemma-3-27b": "google/gemma-3-27b-it",
    "qwen2-72b": "Qwen/Qwen2-72B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1"
} 