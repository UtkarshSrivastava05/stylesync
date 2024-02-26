from flask import Flask, render_template, request, redirect, flash
import os
import io
from PIL import Image
import time

# Additional libraries/functions (optional)
import uuid
from werkzeug.utils import secure_filename
allowed_extensions = ['jpg', 'jpeg', 'png', 'gif']  # List of allowed image extensions


app = Flask(__name__,
            template_folder='/content/stylesync/templates',
            static_folder='/content/stylesync/static'
      )
app.config['UPLOAD_FOLDER'] = 'stylesync/static'  # Configure upload directory
app.config['STATIC_FOLDER'] = 'stylesync/static'

@app.route('/')
def index():
  return render_template('form.html')  # Replace 'form.html' with your template name


@app.route('/process_form', methods=['POST'])
def process_form():
  text = request.form['text_input']
  photo = request.files['photo_upload']

  # Validate file extension
  if photo.filename and photo.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
    flash('Allowed image types: jpg, jpeg, png, gif', 'error')
    return redirect(url_for('index'))

  # Save the photo with a secure filename
  filename = secure_filename(photo.filename)
  photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

  # Generate the complete image URL (add static route)
  image_url = url_for('static', filename=filename)  
  print("image_url: ", image_url)

  return render_template('loading_2.html', text=text, image_url=image_url)

  # return render_template('redirect_message.html', text=text, image_url=image_url)

@app.route('/generate')
def generate():
    text = request.args.get('text')

    image_url = request.args.get('image_url')
    
    # Get the full file path from the image URL
    image_path = os.path.join(app.config['STATIC_FOLDER'], image_url.split('/')[-1])  # Extract filename from URL

    # Open the image file for reading (replace with your processing logic)
    with open(image_path, 'rb') as f:
        image_data = f.read()

    image = Image.open(io.BytesIO(image_data))

    print("image: ", image)

    time.sleep(10)

    # Display the image using the URL
    return f'<img src="{image_url}" alt="Processed Image">'
    # return render_template('loading_2.html')

if __name__ == '__main__':
  app.run()