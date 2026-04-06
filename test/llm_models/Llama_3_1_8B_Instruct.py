import torch
import transformers

# Note: The official repository ID is "meta-llama/Llama-3.1-8B-Instruct".
# "meta-llama/Meta-Llama-3.1-8B-Instruct" might be an alias or incorrect.
model_id = "meta-llama/Llama-3.1-8B-Instruct"
# print(torch.__version__)

# if torch.cuda.is_available():
#     print("GPU is available")
# else:
#     print("GPU is not available")


def main():
    print(f"Loading model: {model_id}...")
    
    # Determine device and torch_dtype
    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.bfloat16
        print("Using GPU (CUDA)")
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
        print("Using CPU (Warning: This model is large and may be slow on CPU)")

    try:
        pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch_dtype},
            device_map=device_map,
        )

        print("\nChatbot ready! Type 'exit' to quit.\n")
        
        # Initialize conversation with a system prompt
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
        ]

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            messages.append({"role": "user", "content": user_input})

            outputs = pipe(
                messages,
                max_new_tokens=256,
                pad_token_id=pipe.tokenizer.eos_token_id,
            )
            
            response = outputs[0]["generated_text"][-1]["content"]
            print(f"Bot: {response}")
            
            messages.append({"role": "assistant", "content": response})

    except Exception as e:
        print(f"\nError: {e}")
        print("Please ensure you have access to the model at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("and are logged in via 'huggingface-cli login'.")

if __name__ == "__main__":
    main()
