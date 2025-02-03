import tensorflow as tf
import matplotlib.pyplot as plt

# Define dataset paths
train_dir = "datasets/UrbanClassification/sky_classification_export/images/train"
val_dir = "datasets/UrbanClassification/sky_classification_export/images/val"

# Load dataset
batch_size = 32
img_size = (224, 224)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=img_size
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=img_size
)

# Get class names
class_names = train_dataset.class_names
print(f"Class Names: {class_names}")

resnet_model = tf.keras.models.Sequential()

pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3),
    pooling='avg',
    classes=4,
    name="resnet50",
)
for layer in pretrained_model.layers:
    layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(tf.keras.layers.Flatten())
resnet_model.add(tf.keras.layers.Dense(512, activation='relu'))
resnet_model.add(tf.keras.layers.Dense(4, activation='softmax'))
#resnet_model.summary()

resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 10
history = resnet_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)
resnet_model.save("urban_resnet_classifier.keras")
