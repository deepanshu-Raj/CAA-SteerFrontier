import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

model = model.to("cuda")
model.eval()

print("Model loaded.")
print("GPU:", torch.cuda.get_device_name(0))

prompt = "[INST] What is the capital of France? [/INST]"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
    )

print("\nOUTPUT:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))