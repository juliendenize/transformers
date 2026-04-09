import torch
from transformers import Mistral3ForConditionalGeneration, AutoProcessor

model_id = "mistralai/Mistral-Medium-3.5-128B"

processor = AutoProcessor.from_pretrained(model_id)
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    mistral_format=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Example: text + image generation
from io import BytesIO
from PIL import Image
import httpx

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
with httpx.stream("GET", url) as response:
    image = Image.open(BytesIO(response.read()))

messages_orig = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "how are you ? What is on this image ?"},
            {"type": "image", "image": image},
        ],
    }
]

print(type(processor))
print(type(processor.tokenizer))
inputs = processor.apply_chat_template(messages_orig, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
print(inputs)
generate_ids = model.generate(**inputs, max_new_tokens=200)
print(generate_ids)
output = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
print(output)
