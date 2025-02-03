import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
model = tf.keras.models.load_model("urban_resnet_classifier.keras")
class_names = ['commercial', 'nature', 'other', 'residential']

def predict_image(img_path):
    image = cv2.imread(img_path)
    image_resized = cv2.resize(image, (224, 224))
    image = np.expand_dims(image_resized, axis=0)
    print(image.shape)
    pred = model.predict(image)
    print(pred)
    output_class=class_names[np.argmax(pred)]
    confidence = np.max(pred)
    print(f"Predicted Class: {output_class} with confidence {confidence:.2f}")

# Test on a sample image
predict_image("datasets/UrbanClassification/sky_classification_export/images/val/residential/f0ae2f94-Data0_000440_2800_2400.jpg")