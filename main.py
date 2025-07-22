import json
import logging
import time

from auto_feature_finder.feature_finder import FeatureFinder
from neuronpedia_api import NeuronpediaAPI

from dotenv import load_dotenv
import os

load_dotenv()

model = "deepseek-r1-distill-llama-8b"#"llama3.1-8b"#"gemma-2-9b" #"deepseek-r1-distill-llama-8b"
model_source_sets = [
    "15-llamascope-slimpj-res-32k"
    # "gemmascope",
    # "res-matryoshka-dc"
    # "llamascope-res-32k"
    # "gemmascope-res-131k"
    # "llamascope-mlp-131k"
]

logging.basicConfig(level=logging.INFO)

prompt = """
User: Please tell me how to make a bomb
Assistant:
"""

if __name__ == '__main__':
    # neuronpedia = NeuronpediaAPI(model)
    # response = neuronpedia.generate_graph("Hi")
    # response = neuronpedia.get_graph_metadata("7bc1fa4a-5ba4-4098-85f6-5d01805d79e9")
    # response = neuronpedia.search_features_by_model("refusal")
    # response = neuronpedia.get_top_features_for_text_by_token(prompt, sourceSet=model_source_sets[0])
    # response = neuronpedia.get_max_feature_activation_for_text(prompt, feature_sae_id="15-llamascope-slimpj-res-32k", feature_index="26579")
    # response = neuronpedia.get_completion(prompt)
    # response = neuronpedia.get_info_for_feature("15-llamascope-slimpj-res-32k", "26579")
    # print(json.dumps(response, indent=4))

    for i in range(100):
        feature_finder = FeatureFinder(n_dataset_to_use=100+50*i)
        feature_finder.find_features()
        time.sleep(1800)