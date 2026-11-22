from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelManager:
    def load_model(self, model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        # Loads on CPU by default; ModelWorker moves the model to its device.
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer
