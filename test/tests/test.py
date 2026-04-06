import torch
import transformers

# Note: The official repository ID is "meta-llama/Llama-3.1-8B-Instruct".
# "meta-llama/Meta-Llama-3.1-8B-Instruct" might be an alias or incorrect.
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
print(torch.__version__)

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
    
