# Real Time Face Prediction
Performs real-time face-based image prediction on gender, age, and facial expression with Neural Networks.

## Description
Gender, age, and facial expression prediction are the critical areas of various real-world applications, such as face recognition applications, discovery about a specific person, and an indication of an individual's intentions within a situation. This project used Deep Convolutional Neural Network such as ResNet50 and modified VGG-16 models for training.

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


### Addition training file:
Available at : this drive [link](https://drive.google.com/drive/folders/1EDQ3PBI6aZ_QVRr0Lj4OYk1c5kRJbkqj?usp=sharing)

This folder includes training data, trained models, and colab notebook from preprocessed data until deploy models from scratch.

## Deploy demo
1. Download prerequisites; the procedure is in **Dependencies** section
2. Run main.py in IDE (for alternative, you can run in the cloud with colab notebook that is listed on **Deploy demo (alternative choice)** section)
3. After the webcam has turned on. On the left upper side, it shows the fps speed of your webcam
4. It will detect your face and predict your gender, age, and facial expression. There are also some additional dummy features.

## Deploy demo (alternative choice)
1. For alternatives, you can execute it on a cloud (google colab) [here](https://colab.research.google.com/drive/1f2uR-2CwUJFSdJrL5ihBLkJXE2Ji-TKq?usp=sharing)
2. Execute ***Preparation*** section
3. Choose which type to execute the demo (Webcam image/video)
4. Input your name and execute it
5. The webcam should turn on by the time

## Sample output:
![image](https://user-images.githubusercontent.com/88226713/162660991-f051d5fe-1f75-48c4-b5fd-c51cf7beaf40.png)


## Notes when running on local:
1. If modules cannot be imported after being downloaded, try clicking Ctrl+Shift+P -> Select python interpreter -> Change to recommended, and restart the IDE
2. If the TensorFlow package cannot be downloaded, try to enable long paths. Open the procedures in this [link](https://superuser.com/questions/1119883/windows-10-enable-ntfs-long-paths-policy-option-missing). In case of long path does not exist, try to enable it with this [link](https://www.thewindowsclub.com/how-to-enable-or-disable-win32-long-paths-in-windows-11-10)
3. On main.py, if the webcam does not work, try changing `cap = cv2.VideoCapture(0)` on line 89 with different numbers like -1 or 1 or 2 instead of 0. Moreover, make sure to allow webcam permissions on the IDE you used.
