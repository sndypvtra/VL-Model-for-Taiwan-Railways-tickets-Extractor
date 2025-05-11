# model_loader.py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def load_model(model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
    """
    Loads the Qwen2.5-VL model and its processor to run exclusively on GPU.
    Raises an error if CUDA (GPU) is not available.
    """
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. This model requires a GPU to run.")

    device = torch.device("cuda")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,   # GPU-compatible precision
        device_map={"": device},     # Force loading entire model to single GPU
        # load_in_4bit=True          # Optional: Uncomment if using quantized model and bitsandbytes installed
    )

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    model.to(device)

    return model, processor, device
