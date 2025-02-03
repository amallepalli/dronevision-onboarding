# dronevision-onboarding

Note: These tasks aren't intended to take too long. Do as good of a job as you think you are capable of with your own brain, a search engine, and the time you have available. 

## Task 1: Use A Pre-Trained Model and Example Code
In the `src` folder, a script called `ultralytics_train.py` has been provided for you. Check out what this file does and what Ultralytics is doing behind the scenes. Run the script (you don't have to run it for many epochs, just get a trained model). Document what you did to get it to work, and provide a brief description of what this code is really doing (take a look at some documentation!). Finally, write a simple script to run the trained model on a user-provided input, and show it works. 

## Task 2: Adapt to a Different Dataset
You'll notice that in the previous example, the dataset was automatically downloaded, and the model architecture was already defined. Now, you will adapt a new dataset to the COCO format and train a new model. Use whichever framework you are most comfortable with.

This step is quite open-ended. First, choose a drone imagery dataset. I've provided a few here but if you can find a nice dataset of urban aerial images, go ahead and use it. 

[UrbanLF](https://github.com/HAWKEYE-Group/UrbanLF/tree/master)

[UrbanScene3D](https://github.com/yilinliu77/UrbanScene3D)

[Urban Classification](https://www.kaggle.com/datasets/fxmikf/aerial-drone-urban-classification)

[Varied Drone Dataset](https://github.com/RussRobin/VDD)

[Semantic Drone Dataset](https://www.kaggle.com/datasets/awsaf49/semantic-drone-dataset/data)

These are the kinds of datasets we will be working with, and working to synthetically augment, in this project team. Note some of these are semantic segmentation, others are classification. 

Next, you will process this dataset, and possibly change the train script, so that you can train a model given the format of the annotations and/or labels that it comes with. It's OK if you change the original YOLO to a semantic segmentation or classification model. **Keep in mind the main point of this exercise is data engineering + being a resourceful programmer, not getting perfect results!**

If you are able to train a model on the dataset you chose, good job! If not, it's OK. Your resourcefulness is what I want to see.

## Deliverable
You need to submit a short (at *most* two pages including pictures and/or code) write-up contaning:
- What you tried to do
- Challenges you faced and how you fixed them
- Results from your work
- Explanation of what you did
- A link to your fork of this repository. 

If you have any questions, please DM `@mgagv` on Discord. Start your message with "ML@P". 