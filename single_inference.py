# single_inference.py
import os
from PIL import Image
import torch
import json
# Import dari file lain
from model_loader import load_model
from parse_ticket import parse_single_ticket_text, add_mileage_to_ticket

# Fungsi Qwen
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Muat model & processor (bisa dilakukan sekali saja)
model, processor, device = load_model()

def process_single_ticket(image_path: str) -> dict:
    """
    1) Cek file
    2) Buka image
    3) Gunakan Qwen => string (teks)
    4) Parse dengan regex => dict
    5) Tambahkan mileage => dict
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File {image_path} not found.")

    image = Image.open(image_path)

    prompt = "請只輸出圖片中的所有文字，不要加入額外敘述或解釋。"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Buat inputs
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        # Hilangkan token input
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    raw_text = output_text[0].strip()

    # Cetak output_text (seluruh list) dan raw_text (string pertama)
    print("Output Text (list):")
    print(json.dumps(output_text, ensure_ascii=False, indent=4))
    print("\nRaw Text (string):")
    print(raw_text)
    # Parse
    parsed_ticket = parse_single_ticket_text(raw_text)
    parsed_ticket = add_mileage_to_ticket(parsed_ticket)
    return parsed_ticket


