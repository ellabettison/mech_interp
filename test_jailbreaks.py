import datasets

from model_calling.Gemini import GeminiLLM
from neuronpedia_api import NeuronpediaAPI


class JailbreakTester:
    def __init__(self):
        self.dataset = datasets.load_dataset("JailbreakBench/JBB-Behaviors")["judge_comparison"]
        self.direct_prompts = self.dataset["goal"]
        self.jailbreak_prompts = self.dataset["prompt"]
        self.model = GeminiLLM()
        self.neuronpedia = NeuronpediaAPI(model="deepseek-r1-distill-llama-8b")
        self.features_to_test = [
            {"eature_sae_id":"15-llamascope-slimpj-res-32k", "feature_index":"26579"}
        ]
        
        # todo auto find denial nodes - nodes which activate only on refusals
    def test_prompt(self, prompt:str):
        # get node activations
        activation = self.neuronpedia.get_feature_activation_for_text(prompt, **self.features_to_test[0])["values"][-1]
        # get refusal
        refusal = self.model.chat("")
        pass
    