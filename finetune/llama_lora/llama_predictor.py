import torch, os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
class LlamaPredictor:
    def __init__(self, tokenizer, model):

        self.tokenizer = tokenizer
        self.model = model

            
    def predict(self, encoded_prompt, temperature=1, max_tokens=1000, stop=None):

        generated_ids = self.model.generate(
            input_ids=encoded_prompt["input_ids"],
            max_new_tokens=max_tokens,
            do_sample=False,
            early_stopping=False,
            num_return_sequences=1,
            temperature=temperature,
            stopping_criteria=stop_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        ).to(device)
        decoded_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return decoded_output[0]