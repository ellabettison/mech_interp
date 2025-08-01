import json
import logging
import os
import uuid
from http.client import HTTPException

import requests


class NeuronpediaAPI:
    def __init__(self, model: str):
        self.api_key = os.environ["NEURONPEDIA_API_KEY"]
        self.model = model

    def generate_graph(self, prompt: str) -> str:
        slug = str(uuid.uuid4())
        response = requests.post(
            "https://www.neuronpedia.org/api/graph/generate",
            headers={"Content-Type": "application/json"},
            json={
                "prompt": prompt,
                "modelId": self.model,
                "slug": slug,
                "maxNLogits": 10,
                "desiredLogitProb": 0.95,
                "nodeThreshold": 0.8,
                "edgeThreshold": 0.85,
                "maxFeatureNodes": 5000,
            },
        )
        logging.getLogger().info(response)
        if response.status_code != 200:
            raise HTTPException(f"Could not generate graph for prompt: {response}")
        return slug

    def get_graph_metadata(self, slug: str):
        response = requests.get(
            f"https://www.neuronpedia.org/api/graph/{self.model}/{slug}"
        )
        return response.content

    def search_features_by_model(
        self, query: str, n_to_generate: int, sourceset_to_test: str
    ):
        features = []
        i = 0
        while len(features) < n_to_generate and i < n_to_generate / 4:
            res = self._search_features_by_model(query, i * 20)
            logging.getLogger().info(f"Got result from search: {len(res)}")
            logging.getLogger().info(
                f"Got result: {[(r['modelId'], r['layer']) for r in res]}"
            )
            filtered_res = [
                r
                for r in res
                if r["modelId"] == self.model and sourceset_to_test in r["layer"]
            ]
            logging.getLogger().info(f"Filtered to {len(filtered_res)}")
            features += filtered_res
            logging.getLogger().info(f"New features size: {(len(features))}")
            i += 1
        return features[:n_to_generate]

    def _search_features_by_model(self, query: str, offset: int = 0):
        """
        Search features by model using provided keyword, returns up to 20 results
        Can set offset param `offset` for pagination
        :param query:
        :return:
        """
        logging.getLogger().info(
            f"Searching for query: {query} and model: {self.model}"
        )
        response = requests.post(
            "https://www.neuronpedia.org/api/explanation/search-model",
            headers={"Content-Type": "application/json"},
            json={
                "modelId": self.model,
                "query": query,
                "offset": offset,
            },
        )
        results = json.loads(response.content)["results"]
        return [self.filter_feature_fields(res) for res in results]

    def get_max_feature_activation_for_text(
        self, text: str, feature_sae_id: str, feature_index: str
    ):
        response = requests.post(
            "https://www.neuronpedia.org/api/activation/new",
            headers={"Content-Type": "application/json"},
            json={
                "feature": {
                    "modelId": self.model,
                    "source": feature_sae_id,
                    "index": feature_index,
                },
                "customText": text,
            },
        )
        print(response)
        print(response.content)
        json_response = response.json()
        if "activations" in json_response:
            return json_response["activations"]["maxValue"]
        return json_response[
            "maxValue"
        ]  # should this be ["activations"]["maxValue"] ??

    def get_feature_activation_for_text_by_token(
        self, text: str, feature_sae_id: str, feature_index: str, token_ind: int = -1
    ):
        response = requests.post(
            "https://www.neuronpedia.org/api/activation/new",
            headers={"Content-Type": "application/json"},
            json={
                "feature": {
                    "modelId": self.model,
                    "source": feature_sae_id,
                    "index": feature_index,
                },
                "customText": text,
            },
        )
        print(response)
        print(response.content)
        json_response = response.json()
        if "activations" in json_response:
            return json_response["activations"]["values"][token_ind]
        return json_response[
            "maxValue"
        ]  # should this be ["activations"]["maxValue"] ??

    def get_info_for_feature(self, layer: str, index: str, filter=True):
        response = requests.get(
            f"https://www.neuronpedia.org/api/feature/{self.model}/{layer}/{index}"
        )
        json_response = json.loads(response.content)

        if filter:
            json_response = self.filter_feature_fields(json_response)

        return json_response

    def filter_feature_fields(self, json_response):
        fields_to_return = [
            "modelId",
            "layer",
            "index",
            "description",
            "neuron",
            "explanations",
        ]
        neuron_fields_to_return = [
            # "neg_str",
            # "pos_str",
            "explanations",
            # "activations"
        ]
        activations_fields_to_return = ["tokens"]
        # Start by filtering the top-level fields
        filtered_json = {
            key: json_response[key] for key in fields_to_return if key in json_response
        }
        # Filter neuron subfields
        if "neuron" in filtered_json and isinstance(filtered_json["neuron"], dict):
            neuron_data = filtered_json["neuron"]
            filtered_json["neuron"] = {
                key: neuron_data[key]
                for key in neuron_fields_to_return
                if key in neuron_data
            }

            # Filter activations sub-subfields
            if "activations" in filtered_json["neuron"]:
                activations = filtered_json["neuron"]["activations"]
                if isinstance(activations, list):
                    filtered_json["neuron"]["activations"] = [
                        {
                            key: item[key]
                            for key in activations_fields_to_return
                            if key in item
                        }
                        for item in activations
                        if isinstance(item, dict)
                    ]
                elif isinstance(activations, dict):
                    filtered_json["neuron"]["activations"] = {
                        key: activations[key]
                        for key in activations_fields_to_return
                        if key in activations
                    }
        return filtered_json

    def get_top_features_for_text(
        self, text: str, sourceSet: str, num_results: int = 100, filter: bool = True
    ):
        response = requests.post(
            "https://www.neuronpedia.org/api/search-all",
            headers={"Content-Type": "application/json"},
            json={
                "modelId": self.model,
                "sourceSet": sourceSet,
                "text": text,
                "selectedLayers": [],
                "sortIndexes": [],
                "ignoreBos": True,
                "densityThreshold": 0.1,
                "numResults": num_results,
            },
        )
        print(response)
        response = response.json()["result"]
        if filter:
            response = [self.filter_feature_fields(feature) for feature in response]
        return response

    def get_top_features_for_text_by_token(
        self, text: str, sourceSet: str, token_ind: int = -1
    ):
        response = requests.post(
            "https://www.neuronpedia.org/api/search-topk-by-token",
            headers={"Content-Type": "application/json"},
            json={
                "modelId": self.model,
                "source": sourceSet,
                "text": text,
                "ignoreBos": True,
                "densityThreshold": 0.1,
                "numResults": 20,
            },
        )
        response = response.json()["results"][token_ind]
        return response["topFeatures"]

    def get_completion(
        self,
        prompt: str,
        layer: str = "15-llamascope-slimpj-res-32k",
        index: str = "26579",
        strength: float = 0,
        temperature: float = 0,
    ):
        response = requests.post(
            "https://www.neuronpedia.org/api/steer",
            headers={"Content-Type": "application/json"},
            json={
                "prompt": prompt,
                "modelId": self.model,
                "features": [
                    {
                        "modelId": self.model,
                        "layer": layer,
                        "index": index,
                        "strength": strength,
                    }
                ],
                "temperature": temperature,
                "n_tokens": 128,
                "freq_penalty": 2,
                "seed": 16,
                "strength_multiplier": 4,
                "steer_special_tokens": False,
            },
        )
        print(response.content)
        return response.json()
