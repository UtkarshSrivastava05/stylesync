# Import necessary libraries
import PIL
import requests
from io import BytesIO
import torch
import cv2
import gradio as gr
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn

# Set the device
device = "cuda"
model_path = "runwayml/stable-diffusion-inpainting"

# Load the stable diffusion model
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to(device)

# Load image processing model for semantic segmentation
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Define label mappings for clothing items
id2label = model.config.id2label
cloth_mapping = {
    'Upper-clothes': ['tshirt', 'shirt', 'top', 'top wear', 'jacket', 'crop top',
                      'sweater', 'cardigan', 'sweatshirt', 'hoodie', 'kurta'],
    'Pants': ['pant', 'trouser', 'jeans', 'leggings'],
    'Dress': ['dress', 'frock', 'one piece', 'long coat', 'jumpsuit']
}

# Function to find the original index based on text prompt and label mappings
def find_original_index(text_prompt, id2label, cloth_mapping):
    for index, label in id2label.items():
        if label in list(cloth_mapping.keys()):
            if any(word in text_prompt.lower() for word in cloth_mapping[label]):
                return index
    return None

# Sample text prompt
prompt = "Change the shirt to a black kurta, with the Spider-Man on it."

# Find the original index based on the prompt
found_index = find_original_index(prompt, id2label, cloth_mapping)

if found_index is not None:
    print(f"Original Index found: {found_index}, Label: {id2label[found_index]}")
else:
    print("No matching label found.")

# Assuming label number 4 is the label you're interested in
desired_label = found_index

# Read and preprocess the image
IMAGE_PATH = '/content/sample_data/white_shirt.jpg'
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
org_img_size = image.shape[0:2]
image = Image.fromarray(image)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits.cpu()

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]

# Create a binary mask for the desired label
mask = np.array(pred_seg == desired_label, dtype=np.uint8)

# Convert binary mask to binary gray (0 or 255)
binary_gray_mask = (mask * 255).astype(np.uint8)

# Create a PIL Image from the binary gray mask
pil_mask = Image.fromarray(binary_gray_mask, mode='L')

# Resize images for processing
image = image.resize((512, 512))
mask_image = pil_mask.resize((512, 512))

# Set parameters for outfit generation
guidance_scale = 7.5
num_samples = 3
generator = torch.Generator(device="cuda").manual_seed(42)

# Generate new outfits based on the prompt and input images
images = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    guidance_scale=guidance_scale,
    generator=generator,
    num_images_per_prompt=num_samples,
).images

# Resize generated images to the original image size
for i in range(len(images)):
    images[i] = images[i].resize(tuple(reversed(org_img_size)))

# Display the image grid
image_grid(images, 1, num_samples + 1)
