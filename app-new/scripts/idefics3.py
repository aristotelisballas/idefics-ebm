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

image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

def run_inference(img_uri, url=False, old_dict=True):
   
    prompt = "User: For the given image, list every visible food item and its food category."

    if url:
        prompts = [[prompt, f"{img_uri}", "<end_of_utterance>"]]
    else:
        img = Image.open(img_uri)
        prompts = [[prompt, img, "<end_of_utterance>"]]

    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(DEVICE)

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_length=100
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    for t in generated_text:
        print(f"{t}\n")

    if old_dict:
        food_groups = find_list_elements_in_string(food_groups_dict, " ".join(generated_text))
    else:
        food_groups = find_list_elements_in_string(food_groups_dict_updated, " ".join(generated_text))

    return food_groups, generated_text
