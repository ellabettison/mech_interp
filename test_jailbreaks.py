import datasets

from model_calling.Gemini import GeminiLLM
from neuronpedia_api import NeuronpediaAPI

# todo !
class JailbreakTester:
    def __init__(self, model: str="deepseek-r1-distill-llama-8b", hf_model: str="fnlp/Llama-Scope-R1-Distill"):
        self.dataset = datasets.load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison")["test"]
        print(self.dataset)
        self.direct_prompts = self.dataset["goal"]
        self.jailbreak_prompts = self.dataset["prompt"]
        self.model = GeminiLLM()
        self.neuronpedia = NeuronpediaAPI(model=model)
        self.features_to_test = [
            {"eature_sae_id":"15-llamascope-slimpj-res-32k", "feature_index":"26579"}
        ]
        self.model = self.load_model()

    def get_completion_for_model(self, prompt: str) -> str:
        return self.model(prompt)
        
        # todo auto find denial nodes - nodes which activate only on refusals
    def test_prompt(self, prompt:str):
        # get node activations
        activation = self.neuronpedia.get_feature_activation_for_text(prompt, **self.features_to_test[0])["values"][-1]
        # get refusal
        refusal = self.model.chat("")
        pass
    
if __name__ == "__main__":
    tester = JailbreakTester()
    res = tester.get_completion_for_model("Hi, how are you?")
    print(res)