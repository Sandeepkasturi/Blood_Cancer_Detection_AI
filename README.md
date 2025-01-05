```
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import gradio as gr

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the pre-trained model
model = tf.keras.models.load_model("maheshbabu.h5")

# Define class labels
classes = ["Normal", "Cancerous"]

# Prediction function
def predict(image):
    try:
        # Load and preprocess the image
        img = load_img(image, target_size=(224, 224))  # Resize image
        img_array = img_to_array(img) / 255.0         # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        # Perform prediction
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])         # Get index of highest probability
        confidence = predictions[0][class_idx]       # Get confidence score

        # Modify the output based on the prediction
        if classes[class_idx] == "Normal":
            return "No Cancer Found"
        else:
            return "Cancer Found!"

    except Exception as e:
        return f"Error processing image: {str(e)}"

# Set up the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath"),  # Use filepath to pass the image path
    outputs="text",  # Change output type to "text" for customized message
    title="Blood Cancer Detection",
    description="Upload an image to detect whether it is Normal or Cancerous."
)

if __name__ == "__main__":
    interface.launch(server_port=7860, server_name="0.0.0.0", share=True)
```

### Key Changes:
1. **Output Message**: The output is customized to print:
   - `"No Cancer Found"` if the model predicts `"Normal"`.
   - `"Cancer Found!"` if the model predicts `"Cancerous"`.
   
2. **Gradio Output Type**: The output type has been changed to `"text"` for displaying the custom message directly.

This will show a message based on the model's prediction, either confirming the presence of cancer or indicating that no cancer was found.


Download the Model from Hugging Face: https://huggingface.co/spaces/skavtech/Blood_Cancer_detection_model/resolve/main/blood_cancer_model.h5
