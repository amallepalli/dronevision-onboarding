from ultralytics import YOLO
import cv2
#load the model that we got from the ultralytics_train script
model = YOLO("runs/detect/train5/weights/best.pt")
#user input for the image they would like to see the model perform on
image_path = input("Enter the path to the image: ")
#model prediction
results = model.predict(image_path, save=True, show=True)
#show the output
for result in results:
    img = result.plot()
    cv2.imshow("Model detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()