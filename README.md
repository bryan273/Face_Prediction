# Real Time Face Prediction
Performs real-time face based image prediction on gender, age, and facial expression.

## Description
Gender, age, and facial expression prediction are the key areas of various real world applications, such as face recognition applications, discovery about a specific person, and indication on an individual's intentions within a situation. This project used Deep Convolutional Neural Network such as ResNet50 and modified VGG-16 models for training.

## Getting Started

### Dependencies

OpenCV | Tensorflow | Numpy | Time | Random
--- | --- | --- |--- |---

Install the required packages by executing the following command.

`$ pip install -r requirements.txt`

### Files
* main.py : main program to execute real-time webcam face prediction
* model:
  * Gender_model.h5 : deep learning model to classify gender
  * Age_model.h5 : deep learning model to estimate age
  * Emotion_model.h5 : deep learning model to predict facial expression (Due to large size of files, download it from this [drive](https://drive.google.com/file/d/1R3H0SCUyd-WVIhU-j6uDqRe6Z8oBqRtr/view?usp=sharing))
  * haarcascade_frontalface_default.xml : detect face bounds


### Addition file:
Available at : this drive [link](https://drive.google.com/drive/folders/1EDQ3PBI6aZ_QVRr0Lj4OYk1c5kRJbkqj?usp=sharing)

This folder includes training data, trained models and colab notebook from preprocessed data until deploy models from scratch

## Deploy demo
1. Download prequisities, procedure is in **Dependencies** section
2. Run main.py in IDE (for alternative you can run in clouds with colab notebooks that is listed on **Addition file** section, make sure to read the README.txt first)
3. After the webcam has turned on. On the left upper side, it shows the fps speed of your webcam
4. It will detected your face and predicted your gender, age, and facial expression. There are also some addition dummy features.

## Sample output:
![image](https://user-images.githubusercontent.com/88226713/162660991-f051d5fe-1f75-48c4-b5fd-c51cf7beaf40.png)


## Notes:
1. If tensorflow package cannot be downloaded, try to enable long paths. Open the procedures in this [link](https://superuser.com/questions/1119883/windows-10-enable-ntfs-long-paths-policy-option-missing). In case of long path doesn't exist, try to enable it with this [link](https://www.thewindowsclub.com/how-to-enable-or-disable-win32-long-paths-in-windows-11-10)
2. On main.py, if the webcam doesnt work, try changing `cap = cv2.VideoCapture(0)` on line 89 with different numbers like -1 or 1 or 2 instead of 0. And make sure to allow webcam permissions on the IDE you used.
