import torch

from steerfrontier.models import load_model


def main():
    model, tokenizer = load_model()

    print("GPU:", torch.cuda.get_device_name(0))
    print("Layers:", model.config.num_hidden_layers)
    print("Hidden size:", model.config.hidden_size)

    prompt = "[INST] What is the capital of France? [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
