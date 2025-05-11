# multi_inference.py
import os
from PIL import Image
import torch

# Import dari file lain
from model_loader import load_model
from parse_ticket import parse_multi_ticket_json, add_mileage_to_ticket

# Fungsi Qwen
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Muat model & processor
model, processor, device = load_model()

def process_multi_ticket(image_path: str):
    """
    1) Cek file
    2) Buka image
    3) Prompt Qwen => JSON
    4) Parse JSON => list of dict
    5) Tambahkan mileage => list of dict
    6) Jika cuma 1 item => return dict, kalau >1 => return list
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File {image_path} not found.")

    image = Image.open(image_path)

    prompt = (
        "你是一個票務信息提取助手，圖片中可能包含多張台鐵或相關票券。"
        "請你依照圖片中票券的呈現順序，逐張票券進行識別，並為每張票券分別提取以下欄位：\n"
        "- date: 日期 (格式為 YYYY.MM.DD)\n"
        "- departure_station: 出發站 (僅輸出拉丁字母/英文形式，請勿保留中文)\n"
        "- arrival_station: 到達站 (同樣僅輸出拉丁字母/英文形式)\n"
        "- departure_time: 出發時間 (格式為 HH:MM)\n"
        "- arrival_time: 到達時間 (格式為 HH:MM)\n"
        "- price: 票價（僅數字，不含貨幣符號）\n\n"

        "如果圖片中有多張票券，請輸出多個 JSON 物件，"
        "並將它們放在一個 JSON 陣列中，保持每張票券資料獨立且不混淆。"
        "請確保無論有幾張票，都按照先後順序（如由左到右、上到下）分開處理。\n\n"

        "請只回傳純粹的 JSON 結構："
        "如果有多張票就輸出如：[{...}, {...}, ...]，"
        "不要包含任何多餘的文字、標籤或格式化符號。"
        "如果某個欄位無法提取，請填空字串。"
        "嚴禁輸出解釋、額外文字或中英文對照。\n\n"

        "票據文本如下："
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Build inputs
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
        generated_ids = model.generate(**inputs, max_new_tokens=320)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    raw_json = output_text[0].strip()

    # Parse multi
    parsed_list = parse_multi_ticket_json(raw_json)

    # Tambahkan mileage
    final_result = []
    for ticket_dict in parsed_list:
        if isinstance(ticket_dict, dict):
            final_result.append(add_mileage_to_ticket(ticket_dict))
        else:
            # Format tidak sesuai, tambahkan apa adanya
            final_result.append(ticket_dict)

    # Jika final_result hanya 1 item => return dict, jika banyak => return list
    if len(final_result) == 1:
        return final_result[0]
    return final_result
