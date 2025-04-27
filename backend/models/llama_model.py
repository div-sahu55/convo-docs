from llama_cpp import Llama

class LlamaModel:
    def __init__(self):
        self.model = Llama(
            model_path="./models/Mistral-7B-Instruct-v0.2.Q4_K_M.gguf",
            n_ctx=4096,   # context window
            n_threads=6,  # optional: adjust based on your CPU cores
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        )

    def query(self, prompt, max_length=1000):
        response = self.model(
            prompt=prompt,
            max_tokens=max_length,
            stop=["<|eot_id|>", "</s>"],  # mistral models usually use <|eot_id|> or </s>
            echo=False
        )
        return response["choices"][0]["text"].strip()
