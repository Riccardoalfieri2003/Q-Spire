from smells.utils.config_loader import get_api_key
from smells.utils.config_loader import get_llm_model
from openai import OpenAI

API_KEY=get_api_key()
LLM_MODEL=get_llm_model()

# Base explainer class with factory pattern
class Explainer:

    _explainers = {}
    
    @classmethod
    def register(cls, smell_class):
        def decorator(explainer_class):
            cls._explainers[smell_class] = explainer_class
            return explainer_class
        return decorator
    
    @classmethod
    def get_explainer(cls, code, smell):
        explainer_class = cls._explainers.get(smell.__class__)
        if explainer_class:
            return explainer_class()  # Return an instance
        return None
    
    @classmethod
    def explain(cls, code, smell):
        """Class method that gets explainer and calls explain in one step"""
        explainer = cls.get_explainer(code, smell)
        if explainer:
            return explainer._explain(code, smell)  # Call the instance method
        return None
    
    def get_prompt(self, code, smell_instance):
        """This should be implemented by each specific explainer"""
        raise NotImplementedError("Subclasses must implement get_prompt method")
    
    def _explain(self, code, smell_instance):
        prompt = self.get_prompt(code, smell_instance)

        #print(prompt)
        #return

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY,
        )

        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True,  # Enable streaming!
        )

        for chunk in completion:
            if chunk.choices[0].delta.content:
                #print(chunk.choices[0].delta.content, end="", flush=True)
                yield chunk.choices[0].delta.content

        smell_instance.explanation=completion