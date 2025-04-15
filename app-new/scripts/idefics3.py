import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from scripts.utils import find_list_elements_in_string, food_groups_dict, food_groups_dict_updated

DEVICE = "cpu"  

checkpoint = "HuggingFaceM4/Idefics3-8B-Llama3"

processor = AutoProcessor.from_pretrained(checkpoint, size={"longest_edge": 364})
processor.image_processor.do_image_splitting = False

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
                {"type": "text", "text": (
                    "What are the food categories and types in the image? List every visible food item along with its specific type and food category. " 
                    "For example, if meat is visible, specify whether it is beef, chicken, etc. "
                    "Be concise but include enough detail to differentiate between similar items."
                )},
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    full_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    parsed_outputs = []
    for output in full_outputs:
        if "Assistant:" in output:
            parsed_output = output.split("Assistant:")[-1].strip()
        else:
            parsed_output = output.strip()
        parsed_outputs.append(parsed_output)
        print(f"{parsed_output}\n")
    
    if old_dict:
        food_groups = find_list_elements_in_string(food_groups_dict, " ".join(parsed_outputs))
    else:
        food_groups = find_list_elements_in_string(food_groups_dict_updated, " ".join(parsed_outputs))
    
    return food_groups, parsed_outputs
