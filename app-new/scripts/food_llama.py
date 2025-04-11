import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from scripts.utils import find_list_elements_in_string, food_groups_dict, food_groups_dict_updated

model_id = "AdaptLLM/food-Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,   
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

def run_inference(img_uri, url=False, old_dict=True):
    if url:
        image = Image.open(requests.get(img_uri, stream=True).raw)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What type of food is shown in this image?"}
                ]
            }
        ]
    else:
        image = Image.open(img_uri)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "List all visible food items and their categories in the image."}
                ]
            }
        ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=50)
    decoded_output = processor.decode(output[0])
    print(decoded_output)
    
    if old_dict:
        food_groups = find_list_elements_in_string(food_groups_dict, " ".join(decoded_output))
    else:
        food_groups = find_list_elements_in_string(food_groups_dict_updated, " ".join(decoded_output))

    return food_groups, decoded_output