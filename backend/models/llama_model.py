from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Model:
    def __init__(self):
        # Load the DistilGPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    def generate_text(self, prompt, max_length=50):
        # Encode the input prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate text
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
