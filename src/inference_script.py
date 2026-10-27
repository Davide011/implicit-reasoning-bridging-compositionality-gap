import torch
from transformers import AutoConfig, GPT2LMHeadModel, AutoTokenizer

# Define the model path
MODEL_PATH = "/scratch/davide/model_paper/outputs_OOD_MODIFIED_composition_SMALL.200.20.18.0/checkpoint-350000/"

# Load the model and tokenizer
config = AutoConfig.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, config=config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#"<e_0><r_14><r_6>"  # is in training , it get it correct out
#"<e_0><r_14><r_3><e_100></a>"   # predict correctly NOT IN TRAININ!!! (however same first hop!)
#<e_0><r_5><r_19><e_37></a>     # IT GET IT CORRECTLY
#"<e_2><r_6><r_7><e_89></a>"  # is in training and it get correct (only e_2 seen)



#<e_2><r_6><r_16><e_137></a>   FAIL

# Prepare the input data for prediction



input_texts = ["<e_0><r_5><r_19>"]
inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)

# Perform inference
model.eval()
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=3,  # Adjust the max_length as needed
        num_return_sequences=1,
    )

# Decode the outputs
predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Print the predictions
for i, prediction in enumerate(predictions):
    print(f"Input: {input_texts[i]}")
    print(f"Prediction: {prediction}")