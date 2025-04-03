import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cpu"
processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3", torch_dtype=torch.bfloat16).to(DEVICE)


# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")


processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
).to(DEVICE)


# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What are the food items and their categories in this image?"},
        ]
    },
]

prompts = [processor.apply_chat_template([message], add_generation_prompt=True) for message in messages]
images = [[image1]]
inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)


def run_inference(img_uri, url=False,old_dict=True):
    if url:
        prompts = [
            [
                "User: Can you identify the food items and their categories in this image?",
                f"{img_uri}",
                "<end_of_utterance>",
            ],
        ]
    else:
        img = Image.open(img_uri)
        prompts = [
            [
                "User: Can you identify the food items and their categories in this image?",
                img,
                "<end_of_utterance>",
            ],
        ]

    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    for t in generated_text:
        print(f"{t}\n")

    # Using the detailed food groups dictionary for more specific matching
    if old_dict:
        food_groups = find_list_elements_in_string(food_groups_dict, " ".join(generated_text))
    else:
        food_groups = find_list_elements_in_string(food_groups_dict_updated, " ".join(generated_text))

    return food_groups, generated_text