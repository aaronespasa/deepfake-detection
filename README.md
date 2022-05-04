# DeepFake Detection with PyTorch 🧐

## Getting Started 🛠
🗂 Clone the repository (the command below uses HTTPS):
```sh
$ git clone https://github.com/aaronespasa/deepfake-detection.git
```

🌲 Create a virtual environment and activate it (make sure you're using Python 3.9):
```sh
$ python3 -m venv ./venv
```
- To activate it in a machine using unix (MacOS or Linux):
```sh
$ source ./venv/bin/activate
```

- To activate it in a machine using Windows:
```sh
$ .\venv\Scripts\activate
```

📄 Install the required libraries:
```sh
$ pip install -r requirements.txt
```

🎉 Now, you are ready to go!

## Roadmap

- [ ] Extract image frame from the videos.
- [ ] Use a MTCNN to detect the faces and create a new dataset.
- [ ] Data Augmentation.
- [ ] Weights & Biases integration.
- [ ] Binary Image Classifier of DeepFakes using a non-SOTA architecture (ex.: InceptionV3 or ResNet50).
- [ ] Binary Image Classifier of DeepFakes using a SOTA architecture (ex. Vision Transformers).
- [ ] Evaluate the image classifier model using Class Activation Maps.
- [ ] Model Deployment (for images) using Streamlit.
- [ ] Write an article describing the project.
- [ ] Binary Video Classifier of DeepFakes.
- [ ] Evaluate the video classifier model using Class Activation Maps.
- [ ] Model Deployment (for videos) using Streamlit.
- [ ] Write an article describing how to improve the binary image classifier to work with video.
- [ ] Binary Classifier for DeepFakes using audio (implementing a Transformer architecture)
- [ ] Model evaluation.
- [ ] Model deployment using PyTorch Live.
- [ ] Write an article describing how to improve the binary video classifier to work with audio.
- [ ] Write a tutorial describing how to do the project.

## Dataset

[FaceForensics](http://niessnerlab.org/projects/roessler2018faceforensics.html): A Large-scale Video Dataset for Forgery Detection in Human Faces.

[FakeCatcher](http://cs.binghamton.edu/~ncilsal2/DeepFakesDataset/): Dataset of synthesized images for deepfake detection.

[Kaggle Dataset augmented by Meta](https://ai.facebook.com/datasets/dfdc/): Dataset from the kaggle competition with more resources provided by Meta.

