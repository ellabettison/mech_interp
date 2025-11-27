import logging
import os

from google import genai
from google.genai import types
import time

from model_calling.LLM import LLM


class GeminiLLM(LLM):
    def __init__(self, temperature=0.0):
        api_key1 = os.environ["GEMINI_API_KEY"]
        self.client = genai.Client(api_key=api_key1)
        self.model = 'gemini-2.0-flash-001' #'gemini-2.0-flash-001'#'gemini-2.5-flash-preview-05-20'
        self.temperature = temperature

    def call_model(self, user_prompt:str, system_prompt:str=None, max_tokens=2_000):
        response = None
        retries = 3
        while retries > 0 and response is None:
            try:
                response = self.client.models.generate_content(
                    model= self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=self.temperature,
                        system_instruction=system_prompt,
                        # thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)
                    )
                )
                logging.getLogger().debug(response)
            except Exception as e:
                logging.getLogger().info(e)
                time.sleep(30)
        return response.text if response is not None else ""