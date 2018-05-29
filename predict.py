import keras
import numpy as np
np.random.seed(420)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model,Model
import cv2
import matplotlib.pyplot as plt
import os
import math
import shutil
import random

nframes = int(40)
#remove temporary files from the directory
os.system('find ./NNFL -regextype awk -regex ".*(\.DS_Store|gt|jpeg|Thumbs|\.jpg).*" > tempfile')
f = open('tempfile', 'r')
b = f.readlines()
f.close()
for x in b:
    os.system('rm -rf ' + x.strip())
os.mkdir('./Frame_folder')
'''
for i in os.listdir('./NNFL'):
    os.mkdir('./Frame_folder/' + i)
    old = os.path.join('./NNFL', i)
    nw = os.path.join('./Frame_folder/' + i)
    for x in os.listdir(old):
        os.mkdir(os.path.join(nw, x))
'''
#get the path of all the movie files in the directory
os.system('find ./NNFL -regextype awk -regex ".*\.avi" > tempfile')
# Playing video from file:
f = open('tempfile', 'r')
b = f.readlines()
f.close()
for x in b:
    x = x.strip()
    cap = cv2.VideoCapture(x)
    currentFrame = 0
#    print(x[:-4].replace('NNFL', 'Frame_folder', 1), "creating")
    os.mkdir(x[:-4].replace('NNFL', 'Frame_folder', 1))
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret: break
        # Saves image of the current frame in jpg file

        name = x[:-4].replace('NNFL', 'Frame_folder', 1) +"/"+str(currentFrame).zfill(4) + '.jpg'
#        print ('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()

def inc(path, le):
    n = nframes - le
    names = sorted(os.listdir(path))
    copies = [int(0) for x in range(le)]
    while (n != 0):
        r = random.randint(0, le - 1)
        shutil.copy(os.path.join(path, names[r]), os.path.join(path, names[r][:-4] + "." + str(copies[r] + 1) + ".jpg"))
        copies[r] += 1
        n -= 1
def red(path, le):
    n = le - nframes 
    names = sorted(os.listdir(path))
    while n > 0:
        if n == 1:
            r = random.randint(0, le - 1)
            os.remove(os.path.join(path, names[r]))
            break
        d = math.ceil(le/(n - 1))
        arr = []
        for i in range(0, le, d):
            arr.append(names[i])
            os.remove(os.path.join(path, names[i]))
            n -= 1
            if (n == 0):
                break
        le -= len(arr)
        for x in arr:
            ind = int(0)
            for i in range(len(names)):
                if names[i] == x:
                    ind = i
                    break
            del names[ind]

Directory = "./Frame_folder/" 
for x in os.listdir(Directory):
    x = Directory + x
    nfiles = len(os.listdir(x))
    if nfiles > nframes:
        red(x, nfiles)
    elif nfiles < nframes:
        inc(x, nfiles)

model = load_model('cnn_final.h5')
classes=[]
X = []
movie_names = []
#Y = []
one_hot = [[1,0,0,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0,0,0,0,0],
[0,0,0,0,1,0,0,0,0,0,0,0],
[0,0,0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,0,0,1,0,0,0,0,0],
[0,0,0,0,0,0,0,1,0,0,0,0],
[0,0,0,0,0,0,0,0,1,0,0,0],
[0,0,0,0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,0,0,0,0,1]]
def loop(path):
    for i,x in enumerate(sorted(os.listdir(path))):
            foo = os.path.join(path, x)
            video = []
            for z in sorted(os.listdir(foo)):
                p = os.path.join(foo, z)
                #print(p, "is p")

                img = cv2.imread(p)
                #cv2.imshow(img)
                
                img = cv2.resize(img,(128,128))
                
                img = np.reshape(img,[1,128,128,3])
                
                l=K.function([model.layers[0].input],[model.layers[10].output])
                out=l([img])[0]
                video.append(np.asarray(out))
            X.append(np.asarray(video))
            movie_names.append(x.strip())

loop('./Frame_folder/')

X = np.asarray(X)
X = np.asarray(X)
np.save('X_test', X)
model1 = load_model('rnn_final2.h5')
X=X.reshape(X.shape[0],X.shape[1],X.shape[3])
Y=model1.predict_classes(X)
d={}
d[0]='golf'
d[1]='kicking'
d[2]='lifting'
d[3]='riding-horse'
d[4]='running'
d[5]='skateboarding'
d[6]='swing-bench'
d[7]='swing-sideangle'
d[8]='walking'
for i in range(len(Y)):
	print (movie_names[i],d[Y[i]])