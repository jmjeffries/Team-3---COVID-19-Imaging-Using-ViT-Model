import os
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import ImageUploadForm 
from .models import UploadImage
import tensorflow as tf
from PIL import Image
import numpy as np
from vit import get_model

# Render the home page
def home(request):
    return render(request, 'home.html')

# Handle image upload and prediction
def upload_image(request):
    context = {}
    if request.method == 'POST':  
        form = ImageUploadForm(request.POST, request.FILES) 
        if form.is_valid():  
            name = form.cleaned_data.get("name")
            img = form.cleaned_data.get("image") 

            # Change the file name to the given name + number of files in the folder
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded')
            num_images = len(os.listdir(upload_dir))
            ext = img.name.split('.')[-1]
            new_filename = f"{name}_{num_images + 1}.{ext}"
            img.name = new_filename

            # Save the image
            image_obj = UploadImage.objects.create(
                name=name, 
                img=img
            )
            image_obj.save()
            img_path = image_obj.img.path
            
            # Analyze the image and process predictions
            predictions = analyze_image(img_path)
            predictions = process_predictions(predictions)

            # Save predictions in session or pass to result page
            request.session['predictions'] = predictions  # Store predictions in session
            
            # Redirect to result page
            return redirect('result')  # Redirect to 'result' view

    else:
        form = ImageUploadForm()
    context['form'] = form
    return render(request, 'home.html', context)

def login(request):
    return render(request, 'login.html')

# Display result page with predictions
def result(request):
    predictions = request.session.get('predictions', [])  # Get predictions from session
    return render(request, 'result.html', {'predictions': predictions})

# Analyze image and make predictions using the model
def analyze_image(img_path):
    model = get_model()  # Load the model
    img = Image.open(img_path)

    # Resize image to match model input shape 
    img = img.resize((72, 72))

    # Convert image to grayscale
    img = img.convert('L')

    img_array = np.array(img)

    # Reshape the image for the model
    img_array = img_array.reshape(1, 72, 72, 1)  # Adjusting shape
    img_array = img_array / 255.0  # Normalize pixel values

    # Predict using the model
    predictions = model.predict(img_array)

    return predictions

# Process the raw predictions from the model
def process_predictions(predictions):
    print("Preprocess Prediction:", predictions)
    # Apply softmax to convert logits to probabilities
    softmax_output = tf.nn.softmax(predictions).numpy()

    # Get the index of the predicted class
    predicted_class = np.argmax(softmax_output)

    print("Predicted class index:", predicted_class)


    # COVID or not
    if predicted_class == 0:
        result = "COVID Free"
    else:
        result = "COVID LIKELY"

    return [result]  # Return the result as a list so it's iterable in the template
