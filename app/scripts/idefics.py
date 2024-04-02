import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image
from scripts.utils import find_list_elements_in_string, food_groups_dict

device = "cpu"
checkpoint = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)

def run_inference(img_uri, url=False):
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
    food_groups = find_list_elements_in_string(food_groups_dict, " ".join(generated_text))
    return food_groups, generated_text



def test_endpoint(img_url):
    return img_url

