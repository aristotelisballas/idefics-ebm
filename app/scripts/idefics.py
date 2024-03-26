import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

from scripts.utils import find_list_elements_in_string, MAJOR_FOOD_GROUPS
#from ebm_idefics.app.utils import find_list_elements_in_string, MAJOR_FOOD_GROUPS

from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

checkpoint = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)


# "https://media.cnn.com/api/v1/images/stellar/prod/190715182611-19-germany-dishes-gallery-restricted.jpg?q=w_2700,h_1800,x_0,y_0,c_fill/h_618",


def run_inference(img_uri, url=False):
    # We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.

    if url:
        prompts = [
            [
                "User: What are the food groups in this image?",
                f"{img_uri}",
                "<end_of_utterance>",

                # "User: What are the food groups in this image?",
                # img_url,
                # "<end_of_utterance>",
            ],
        ]
    else:
        img = Image.open(img_uri)
        prompts = [
            [
                # "User: What types of food are in this image?",
                # f"{img_url}",
                # "<end_of_utterance>",

                "User: What are the food groups in this image?",
                img,
                "<end_of_utterance>",
            ],
        ]

    # --batched mode
    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
    # --single sample mode
    # inputs = processor(prompts[0], return_tensors="pt").to(device)

    # Generation args
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    for i, t in enumerate(generated_text):
        print(f"{i}:\n{t}\n")

    food_groups = find_list_elements_in_string(MAJOR_FOOD_GROUPS, generated_text)

    return food_groups, generated_text


def test_endpoint(img_url):
    return img_url

