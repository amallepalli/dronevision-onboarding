import os
import json
import shutil
import random
def organize_images(json_file, base_dir, train_ratio):
    with open(json_file, "r") as f:
        data = json.load(f)

    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    #creates a folder for each type of classification. combining the classifications "roads" and "other".
    categories = set()
    for entry in data:
        if "choice" in entry:
            category = entry["choice"]
            if category in ["other", "roads"]:
                category = "other"

            categories.add(category)

    for category in categories:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)

    category_images = {category: [] for category in categories}
    #moving the images to the right folder
    for entry in data:
        img_path = base_dir + "/" + entry["image"][7:]
        if "choice" not in entry:
            continue
        
        if entry["choice"] in ["other", "roads"]:
            choice = "other"
        else:
            choice = entry["choice"]

        if os.path.exists(img_path):
            category_images[choice].append(img_path)
        else:
            print(f"{img_path} not found!")
    
    for category, images in category_images.items():
        random.shuffle(images)
        split_num = int(len(images) * train_ratio)

        train_images, val_images = images[:split_num], images[split_num:]

        for img in train_images:
                shutil.move(img, os.path.join(train_dir, category, os.path.basename(img)))
        for img in val_images:
                shutil.move(img, os.path.join(val_dir, category, os.path.basename(img)))
    print("Finished organizing data")

json_file = "datasets/UrbanClassification/sky_classification_export/sky_classification_export.json"
base_dir = "datasets/UrbanClassification/sky_classification_export/images"
train_ratio = 0.8
organize_images(json_file, base_dir, train_ratio)
