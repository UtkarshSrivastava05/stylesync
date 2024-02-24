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
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import os

app = Flask(_name_)

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

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[1].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

app = Flask(__name__,
            template_folder='/content/stylesync/templates',
            static_folder='/content/stylesync/static'
      )

@app.route("/")
def home():
    return render_template("index_new.html")

@app.route('/generate', methods=['POST'])
def generate():
    # Access the uploaded file and prompt from the request
    photo = request.files['photo']
    prompt = request.form['text_prompt']

    # Redirect to the loading page while processing
    return redirect(url_for('loading'))

    print('prompt first',prompt)

    if photo:
        # Save the uploaded file to a temporary location
        filename = secure_filename(photo.filename)
        file_path = os.path.join('/', filename)
        photo.save(file_path)

        # Open the file using PIL
        image = Image.open(file_path)

        # Now you can perform any operations on the image using PIL
        # For example, you can display it
        # image.show()
    print('type of photo',type(image))
    print('photo printing',image)

    # Sample text prompt
    # prompt = "Change the shirt to a black kurta, with the Spider-Man on it."
    # prompt = request.form('text_prompt')
    # print('prompt first',prompt)

    # if 'photo' in request.files:
    #     image = request.files['photo']


    # f = request.files['file']

    # # Save the file to ./uploads
    # basepath = os.path.dirname(_file_)
    # file_path = os.path.join(
    #     basepath, 'uploads', secure_filename(f.filename))
    # f.save(file_path)

    # print('prompt',prompt)
    # print('image',image)

    # # Find the original index based on the prompt
    found_index = find_original_index(prompt, id2label, cloth_mapping)

    if found_index is not None:
        print(f"Original Index found: {found_index}, Label: {id2label[found_index]}")
    else:
        print("No matching label found.")

    # # Assuming label number 4 is the label you're interested in
    desired_label = found_index

    # Read and preprocess the image
    IMAGE_PATH = file_path

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

    image_org = image.resize(tuple(reversed(org_img_size)))

    # insert initial image in the list so we can compare side by side
    images.insert(0, image_org)
    
    # Resize generated images to the original image size
    for i in range(len(images)):
        images[i] = images[i].resize(tuple(reversed(org_img_size)))

    # Display the image grid
    generated_images = image_grid(images, 1, num_samples + 1)

    # Redirect to the result page
    return render_template("result.html")


    # return jsonify({'status': 'success', 'message': 'Image and prompt received successfully'})

# @app.route('/upload_image',methods=['POST'])
# def upload_image():
#     # Get the uploaded file
#     uploaded_file = request.files['image']

#     return render_template('index_new.html', prediction_text='Employee image should be $ {}'.format(output))


# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

# Add routes for the loading and result pages
@app.route('/loading')
def loading():
    return render_template("loading.html")

@app.route('/result')
def result():
    # You may need to pass dynamic data to the result template, e.g., image path
    image_path = url_for('static', filename='result.jpg')
    return render_template("result.html", image_path=image_path)

if __name__ == "__main__":
    app.run()