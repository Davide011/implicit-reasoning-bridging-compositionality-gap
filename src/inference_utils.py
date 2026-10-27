import torch
from transformers import AutoConfig, GPT2LMHeadModel, AutoTokenizer

def load_model_and_tokenizer(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_predictions(model, tokenizer, input_texts, device, max_length=50, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95, stop_pattern="</a>"):
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,  # Enable sampling
        )
    
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    def post_process_prediction(prediction, stop_pattern):
        stop_index = prediction.find(stop_pattern)
        if stop_index != -1:
            return prediction[:stop_index + len(stop_pattern)]
        return prediction
    
    processed_predictions = [post_process_prediction(prediction, stop_pattern) for prediction in predictions]
    return processed_predictions

def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")