# Import needed tools
from tensorflow import keras
import numpy as np
import cv2
import time
import random

# Crop Face
cascade_face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def crop_resize(img, face_box , x_max , y_max):
  x, y, w, h = face_box

  # Wider the crop 15 px each and 25 px for neck
  crop_img = img[(y-25 if y-25>0 else 0):(y+h+15 if y+h+15<y_max else y_max) , 
            (x-15 if x-15>0 else 0):(x+w+15 if x+w+15<x_max else x_max)]
  resized_img = cv2.resize(crop_img, (256, 256)) # Resize the picture resolution

  return resized_img

#Preprocessing image

def preprocess(img, out=True):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = cascade_face_detector.detectMultiScale(gray, scaleFactor=1.5)
  print(faces)
  if out: print(faces) 
  cropped_resized_img = crop_resize(img, faces[0], img.shape[0],img.shape[1]) # Crop First detected image
  return cropped_resized_img, faces[0]

model_1 = keras.models.load_model('Gender_clf_model.h5')
model_2 = keras.models.load_model('Age_rgr_model.h5')
model_3 = keras.models.load_model('Emotion_model.h5')

# Predict whether image gender is a man or woman and age estimation

def predict_img(model_1,model_2,model_3, img):

    emoji = {1:('Flat','Dont forget','to smile :)'), 2:('Happy','Glad you ','smile :D'), 3:('Sad','Why are','you sad :('), 4:('Mad','Jangan marah','marah ><')}

    # Load and preprocess image
    img,box = preprocess(img, False)
    img = img/255

    # Convert image to array
    x = keras.preprocessing.image.img_to_array(img)
    images = np.expand_dims(x, axis=0)

    # Predict image
    classes = model_1.predict(images, batch_size=10)
    predicted_gender = ("Male" if np.argmax(classes) == 1 else "Female")
    [[age]] = model_2.predict(images, batch_size=10)
    emotion = model_3.predict(images, batch_size=10)
  
    if predicted_gender == 'Male':
      return 'Male', round(classes[0][1]*100,2) , round(age) , emoji[np.argmax(emotion)+1], box
    else:
      return 'Female', round(classes[0][0]*100,2) , round(age) , emoji[np.argmax(emotion)+1], box

def Recognition(name):
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    while True:

        # 1. Import image
        _, img = cap.read()
      
        # create transparent overlay for bounding box
        font = (cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0) , 2)
        try:
            gender ,conf, age, emotion, box = predict_img(model_1,model_2,model_3, img)
            left, top, width, height = box
        except IndexError:
            text = "No Face Detected"
            cv2.putText(img, text, (200,50), *font)
            cv2.imshow("Image", img)
            if cv2.waitKey(20) & 0xFF == ord('q'):
              break
            continue
            
        font_1 = (cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0) , 2)
        cv2.rectangle(img, (left, top), (left+width, top+height+20), (0,255,0) , 2)
        cv2.putText(img, name,(left+10, top+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(176,224,230) , 2)
        
        cv2.putText(img, f"{gender}, Age: {age}",(left, top - 53), *font_1)
        cv2.putText(img, f"Mood: {emotion[0]}",(left, top - 30), *font_1)
        cv2.putText(img, f"({conf}%)  ",(left, top - 10), *font_1)
        
        cv2.putText(img, f"Cuteness  : {random.randint(96,99)}% ",(left, top + height + 37), *font_1)
        cv2.putText(img, f"Kindness  : {random.randint(94,98)}% ",(left, top + height + 57), *font_1)
        cv2.putText(img, f"Attractive  : {random.randint(95,99)}% ",(left, top + height + 77), *font_1)
        cv2.putText(img, emotion[1],(left+10, top+height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(176,224,160) , 2)
        cv2.putText(img, emotion[2],(left+10, top+height+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(176,224,160) , 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        font_2 = cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
        cv2.rectangle(img, (0, 0), (160, 80), (0, 50, 255), cv2.FILLED)
        cv2.putText(img, 'Speed :', (10, 30), *font_2) 
        cv2.putText(img, str(round(fps,2)), (10, 70), *font_2) 

        cv2.imshow("Image", img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
          break
    cap.release()
    cv2.destroyAllWindows()


Recognition("Bryan")

