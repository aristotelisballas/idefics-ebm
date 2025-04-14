import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from scripts.utils import find_list_elements_in_string, food_groups_dict, food_groups_dict_updated

DEVICE = "cpu"  

checkpoint = "HuggingFaceM4/Idefics3-8B-Llama3"

processor = AutoProcessor.from_pretrained(checkpoint, size={"longest_edge": 4 * 364})
model = AutoModelForVision2Seq.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(DEVICE)

def run_inference(img_uri, url=False, old_dict=True):
    if url:
        image = load_image(img_uri)
    else:
        image = Image.open(img_uri)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
              {"type": "text", "text": "What are the food categories and types in the image? Respond concisely without extra explanations. Only list the items and their categories."},
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    generated_ids = model.generate(**inputs, max_new_tokens=150)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    for t in generated_text:
        print(f"{t}\n")
    
    if old_dict:
        food_groups = find_list_elements_in_string(food_groups_dict, " ".join(generated_text))
    else:
        food_groups = find_list_elements_in_string(food_groups_dict_updated, " ".join(generated_text))
    
    return food_groups, generated_text
