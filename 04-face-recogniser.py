#!/usr/bin/env python
"""

You need to place a series of face images in a subfolder of faces for this to work.

Some code is borrowed from the BSD licenced example work of Philipp Wagner. (See the 'borrowed' directory
for my cleaned up version of his work.

"""
import freenect
import cv
import os
import sys
import cv2
import numpy as np


def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X -= float(minX)
    X /= float((maxX - minX))
    # scale to [low...high].
    X = X * (high - low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


def read_images(path, sz=None):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)

                    if im is None:
                        continue
                        # resize to given size (if given)
                    if sz is not None:
                        im = cv2.resize(im, sz)

                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c += 1

    return [X, y]


def video_to_bgr(video):
    video = video[:, :, ::-1]  # RGB -> BGR
    image = cv.CreateImageHeader((video.shape[1], video.shape[0]), cv.IPL_DEPTH_8U, 3)
    cv.SetData(image, video.tostring(), video.dtype.itemsize * 3 * video.shape[1])
    return image


def show_faces():
    image = video_to_bgr(freenect.sync_get_video()[0])
    min_size = (20, 20)
    image_scale = 2
    haar_scale = 1.2
    min_neighbors = 2
    haar_flags = 0

    gray = cv.CreateImage((image.width, image.height), 8, 1)
    small_image = cv.CreateImage((cv.Round(image.width / image_scale), cv.Round(image.height / image_scale)), 8, 1)
    cv.CvtColor(image, gray, cv.CV_BGR2GRAY)
    cv.Resize(gray, small_image, cv.CV_INTER_LINEAR)
    cv.EqualizeHist(small_image, small_image)

    faces = cv.HaarDetectObjects(
        small_image,
        face_cascade,
        cv.CreateMemStorage(0),
        haar_scale,
        min_neighbors,
        haar_flags,
        min_size
    )

    if faces:
        for face in faces:
            [X, y] = read_images('faces/', (small_image.width, small_image.height))
            y = np.asarray(y, dtype=np.int32)
            model = cv2.createEigenFaceRecognizer()
            model.train(np.asarray(X), np.asarray(y))

            [p_label, p_confidence] = model.predict(np.fromstring(small_image.tostring(), dtype=np.uint8))
            # Print it:
            print "Predicted label = %d (confidence=%.2f)" % (p_label, p_confidence)

    cv.ResetImageROI(image)

    return image

if __name__ == "__main__":

    # Build named Window
    cv.NamedWindow('Faces')
    # Load Haar Cascade
    face_cascade = cv.Load('haar/frontalface.xml')

    while True:
        cv.ShowImage('Faces', show_faces())

        if cv.WaitKey(10) == 27:
            break
