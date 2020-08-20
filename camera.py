import cv2
import cv2
import numpy as np
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
import tensorflow as tf

from fr_utils import *
from inception_blocks_v2 import *
threshold = 0.25



face_database = {}

FR_model = load_model('nn4.small2.v1.h5')
#print("Total Params:", FR_model.count_params())

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6




for name in os.listdir('images'):
	for image in os.listdir(os.path.join('images',name)):
		identity = os.path.splitext(os.path.basename(image))[0]
		face_database[identity] = fr_utils.img_path_to_encoding(os.path.join('images',name,image), FR_model)

print(face_database)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            roi = image[y:y + h, x:x + w]
            encoding = img_to_encoding(roi, FR_model)
            min_dist = 100
            identity = None

            for (name, encoded_image_name) in face_database.items():
                dist = np.linalg.norm(encoding - encoded_image_name)
                if (dist < min_dist):
                    min_dist = dist
                    identity = name
                print('Min dist: ', min_dist)

            if min_dist < 0.1:
                cv2.putText(image, "Face : " + identity[:-1], (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(image, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            else:
                cv2.putText(image, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

                break
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()



