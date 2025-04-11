import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from scripts.utils import find_list_elements_in_string, food_groups_dict, food_groups_dict_updated

model_id = "AdaptLLM/food-Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
).to('cpu')
processor = AutoProcessor.from_pretrained(model_id)

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

# NOTE: For AdaMLLM, always place the image at the beginning of the input instruction in the messages.
# messages = [
#     {"role": "user", "content": [
#         {"type": "image"},
#         {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
#     ]}
# ]
# input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(
#     image,
#     input_text,
#     add_special_tokens=False,
#     return_tensors="pt"
# ).to(model.device)
#
# output = model.generate(**inputs, max_new_tokens=30)
# print(processor.decode(output[0]))

def run_inference(img_uri, url=False,old_dict=True):
    if url:
        image = Image.open(requests.get(img_uri, stream=True).raw)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "What type of food is shown in this image?"}
            ]}
        ]
    else:
        image = Image.open(img_uri)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "For the given image, list every visible food item and its food category."}
            ]}
        ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=30)
    print(processor.decode(output[0]))
    generated_text = processor.decode(output[0])
    # for t in generated_text:
    #     print(f"{t}\n")

    # Using the detailed food groups dictionary for more specific matching
    if old_dict:
        food_groups = find_list_elements_in_string(food_groups_dict, " ".join(generated_text))
    else:
        food_groups = find_list_elements_in_string(food_groups_dict_updated, " ".join(generated_text))

    return food_groups, generated_text