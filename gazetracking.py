import cv2, dlib
import numpy as np
import time, datetime
from imutils import face_utils
from keras.models import load_model
import matplotlib.pyplot as plt

IMG_SIZE = (150, 150)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = load_model('ourmodel2.h5')
model.summary()
def makeTS_03():
    return str(int(datetime.datetime.now().timestamp()))
def convertDate(ts):
    return datetime.datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d %H:%M:%S')
def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

# main
cap = cv2.VideoCapture(0)
front = []
side = []
sumfront =0
sumside =0
index=0
fn=[]
sn=[]
f=0
s=0
n=0
while cap.isOpened():
  ret, img_ori = cap.read()

  if not ret:
    break

  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    cv2.imshow('l', eye_img_l)
    cv2.imshow('r', eye_img_r)
    # 50,150 <-원래값 100,75<-현지값
    eye_input_l = eye_img_l.copy().reshape((1, 100, 75, 3)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, 100, 75, 3)).astype(np.float32) / 255.

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)
    list_l=pred_l.tolist()
    list_r=pred_r.tolist()
    n+=1
    print(n,"번째 프레임")
    print(pred_l)
    print(pred_r)
    front.append(list_l[0][0])
    sumfront+=list_l[0][0]
    f+=1
    fn.append(f)
    side.append(list_l[0][1])
    sumside+=list_l[0][1]
    s+=1
    sn.append(s)
   
    front.append(list_r[0][0])
    sumfront+=list_r[0][0]
    f+=1
    fn.append(f)
    side.append(list_r[0][1])
    sumside+=list_r[0][1]
    s+=1
    sn.append(s)
	
    
	
    ts=makeTS_03()
    dt=convertDate(ts)
    print(dt)   
    if(pred_l.argmax() == 0):
        print('left eye : looking foward')
    else:
        print('left eye : looking side')
    if(pred_r.argmax() == 0):
        print('right eye : looking foward')
    else:
        print('right eye : looking side')  

    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)
 
  cv2.imshow('result', img)
  if cv2.waitKey(1) == ord('q'):
    print('총',len(front),'개 프레임의 정면평균 :',sumfront/len(front))
    print('총',len(side),'개 프레임의 측면평균 :',sumside/len(side)) 
    plt.scatter(fn, front)
    plt.ylabel('predict')
    plt.scatter(sn, side)
    plt.ylabel('predict')
    plt.title('predict')
    plt.show()
    break