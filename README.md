
# Realtime Face Mask Detection

## Description

Object Detection API. The system utilizes transfer learning to train a pre-trained model, specifically the SSD MobileNet, for the task of localizing and detecting face masks. The detection model is trained to recognize two classes: "Mask" and "No Mask."

## Key Features
* Real-time face mask detection.
* Utilizes transfer learning technique for efficient model training.
* TensorFlow Object Detection API is employed for model training and deployment.
* Pre-trained model: SSD MobileNet.
*  Two classes: "Mask" and "No Mask."
## Screenshots


https://github.com/Bytecode-Magnum/Realtime-Face-Mask-Detection/assets/99680514/6f37c795-6a3a-496f-8537-ca7f10cdb517


![Screenshot 2023-12-07 214303](https://github.com/Bytecode-Magnum/Realtime-Face-Mask-Detection/assets/99680514/1ff27a50-8745-4ea7-bf34-55e6082b239d)

![Screenshot 2023-12-07 214357](https://github.com/Bytecode-Magnum/Realtime-Face-Mask-Detection/assets/99680514/6417bae5-03c1-4213-a8c9-2709e8c04fd3)






## How It Works
The system leverages the power of the TensorFlow Object Detection API to facilitate the training and deployment of the face mask detection model. Here's a step-by-step breakdown:

1. Data Annotation with LabelImg:
 * The process begins with annotating a dataset of images using LabelImg. This annotation tool allows for the manual labeling of individuals' faces as either "Mask" or "No Mask," creating a labeled dataset for model training.
2. Transfer Learning with SSD MobileNet:

* The pre-trained SSD MobileNet model serves as the base architecture for face mask detection. Through transfer learning, the model is fine-tuned using the annotated dataset. Transfer learning enables the model to adapt its knowledge from a general object detection task to the specific task of recognizing face masks.

3. Training the Model:
* The annotated dataset is split into training and test sets. The model is then trained on the training set, adjusting its parameters to accurately detect and classify faces with and without masks. The validation set ensures the model's generalization to new, unseen data.

4. TensorFlow Object Detection API:

* The TensorFlow Object Detection API streamlines the implementation of the model, providing tools for training, evaluation, and deployment. It abstracts complex details, making it easier to integrate the trained face mask detection model into real-world applications.
5. Real-Time Detection in Images and Video Streams:

* The resulting model is capable of real-time face mask detection. Whether applied to images or video streams, the system can swiftly and accurately identify individuals wearing masks and those without.

## Installation guide

1. TensorFlow Object Detection API
* To get started with the project, it's essential to set up the TensorFlow Object Detection API. Follow the official documentation for detailed instructions:
[Official TensorFlow Object Detection API Installation Guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/).

Ensure that you have TensorFlow 2.x installed before proceeding.

2. Dataset Preparation with LabelImg
* Before training the face mask detection model, you need to create an annotated dataset using LabelImg. Follow these steps:
* [Install LabelImg by following the instructions on the LabelImg GitHub repository](https://github.com/HumanSignal/labelImg).
* Use LabelImg to annotate your dataset of images, labeling individuals' faces as either "Mask" or "No Mask."

3. Model Training
* Now that you have your annotated dataset, you can proceed with training the face mask detection model. Utilize the provided code to perform transfer learning on the SSD MobileNet base model:

  

To train the model

```bash
  # Clone the project repository
git clone https://https://github.com/Bytecode-Magnum/Realtime-Face-Mask-Detection.git
cd your_project

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows, use 'venv\Scripts\activate'

# Install required dependencies
pip install -r requirements.txt

# Set up the TensorFlow Object Detection API
# Follow the official installation guide if you haven't done this yet

# Prepare the dataset and annotations
# Place your annotated images and corresponding XML files in the 'annotations' and 'images' folders, respectively

# Train the model
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=5000

```
* Feel free to customize the installation guide based on your project's structure, file paths, and any additional details specific to your implementation.




