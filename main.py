# Import needed tools
from tensorflow import keras
import numpy as np
import cv2
import time
import random

# Load model
cascade_face_detector = cv2.CascadeClassifier("model\haarcascade_frontalface_default.xml")
model_1 = keras.models.load_model('model\Gender_model.h5')
model_2 = keras.models.load_model('model\Age_model.h5')
model_3 = keras.models.load_model('model\Emotion_model.h5')

# Crop image for more accurate prediction
def crop_resize(img, face_box , x_max , y_max):
  x, y, w, h = face_box

  # Wider the crop 15 px each sides except 25 px for bottom side
  try:
    crop_img = img[(y-25 if y-25>0 else 0):(y+h+15 if y+h+15<y_max else y_max) , 
              (x-15 if x-15>0 else 0):(x+w+15 if x+w+15<x_max else x_max)]
    resized_img = cv2.resize(crop_img, (256, 256)) # Resize the picture resolution
  except:
    crop_img = img[y:y+h,x:x+w]
    resized_img = cv2.resize(crop_img, (256, 256)) 
  return resized_img

# Preprocess image
def preprocess(img, out=True):
  # Detect face
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = cascade_face_detector.detectMultiScale(gray)

  # Check detected face
  if len(faces) == 0 : raise IndexError
  if out: print(faces) 

  # Cropped detected face
  cropped_resized_img = crop_resize(img, faces[0], img.shape[0],img.shape[1])
  return cropped_resized_img, faces[0]

# Making prediction on faces
def predict_img(model_1,model_2,model_3, img):

    emoji = {1:('Flat','Dont forget','to smile !'), 2:('Happy','Glad you ','are happy'), 3:('Sad','You are a','strong person'), 4:('Mad','Dont be mad','Calm down')}

    # Load and preprocess image
    img,box = preprocess(img, False)
    img = img/255

    # Convert image to array
    x = keras.preprocessing.image.img_to_array(img)
    images = np.expand_dims(x, axis=0)

    # Predict image
    classes = model_1.predict(images, batch_size=10)
    predicted_gender = ("Male" if np.argmax(classes) else "Female")
    [[age]] = model_2.predict(images, batch_size=10)
    emotion = model_3.predict(images, batch_size=10)
    
    if predicted_gender == 'Male':
      return 'Male', round(classes[0][1]*100,2) , round(age) , emoji[np.argmax(emotion)+1], box
    else:
      return 'Female', round(classes[0][0]*100,2) , round(age) , emoji[np.argmax(emotion)+1], box

# Set filler text
def set_text(img, name, gender, conf, age, emotion, box):
    left, top, width, height = box

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

# Check webcam fps
def set_time(img, fps):

    font_2 = cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
    cv2.rectangle(img, (0, 0), (160, 80), (0, 50, 255), cv2.FILLED)
    cv2.putText(img, 'Speed :', (10, 30), *font_2) 
    cv2.putText(img, str(round(fps,2)), (10, 70), *font_2) 

def main(name='Your Name'):
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    cnt=0
    while True:
        _, img = cap.read()     
        img = cv2.flip(img, 1)

        # Re-update every 5 iteration
        if cnt%5:
            cv2.imshow("Image", img)
            cnt+=1
            continue

        # Check detected face
        try:
            gender ,conf, age, emotion, box = predict_img(model_1,model_2,model_3, img)
            set_text(img, name, gender ,conf, age, emotion, box)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            set_time(img, fps)

        except IndexError:
            font = (cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0) , 2)
            text = "No Face Detected"
            cv2.putText(img, text, (200,50), *font)  

        cv2.imshow("Image", img)
        if cv2.waitKey(20) & 0xFF == ord('q'): break
        cnt+=1
        print(cnt)

    cap.release()
    cv2.destroyAllWindows()

main()

