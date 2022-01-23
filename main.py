import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
from tensorflow.keras import models
from get_keys import key_check

import random
model_name = ''
model = models.load_model(model_name)

t_time = 0.09


def straight():
    ##    if random.randrange(4) == 2:
    ##        ReleaseKey(W)
    ##    else:
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    PressKey(W)
    PressKey(A)
    # ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    # ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)


def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    # ReleaseKey(W)
    # ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)


def back():
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def main():
    count = 0
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while True:

        if not paused:
            count = count + 1
            screen = np.array(ImageGrab.grab(bbox=(30, 40, 800, 600)))
            cv2.imwrite(f'images/img{count}.png', screen)
            last_time = time.time()
            screen = cv2.resize(screen, (160, 120))
            images = np.array([screen]).reshape(-1, 120, 160, 3)
            prediction = model.predict(images)[0]
            print(prediction)
            turn_thresh = .75
            fwd_thresh = 0.78

            if prediction[0] > fwd_thresh:
                straight()
            elif prediction[1] > turn_thresh:
                left()
            elif prediction[2] > turn_thresh:
                right()
            else:
                back()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)


main()
