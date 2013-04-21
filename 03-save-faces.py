#!/usr/bin/env python
import freenect
import cv
import random
import time


def video_to_bgr(video):
    video = video[:, :, ::-1]  # RGB -> BGR
    image = cv.CreateImageHeader((video.shape[1], video.shape[0]), cv.IPL_DEPTH_8U, 3)
    cv.SetData(image, video.tostring(), video.dtype.itemsize * 3 * video.shape[1])
    return image


def find_faces(dev, data, timestamp):
    global keep_running
    global last_time
    global has_face
    image = video_to_bgr(data)
    image2 = image
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
        has_face = True
        for ((x, y, w, h), n) in faces:
            pt1 = (int(x * image_scale), int(y * image_scale))
            pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
            cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)
            cv.SetImageROI(image, (pt1[0], pt1[1], pt2[0] - pt1[0], pt2[1] - pt1[1]))
            cv.Copy(image, image2)
            cv.SaveImage('faces/face-%s.png' % n, image2)
            cv.SetImageROI(image, (pt1[0], pt1[1], pt2[0] - pt1[0], int((pt2[1] - pt1[1]) * 0.7)))

        eyes = cv.HaarDetectObjects(
            image,
            eye_cascade,
            cv.CreateMemStorage(0),
            haar_scale,
            min_neighbors,
            haar_flags,
            (15, 15)
        )

        if eyes:
            for eye in eyes:
                cv.Rectangle(
                    image,
                    (eye[0][0], eye[0][1]),
                    (eye[0][0] + eye[0][2], eye[0][1] + eye[0][3]),
                    cv.RGB(255, 0, 0),
                    1,
                    8,
                    0
                )
    else:
        if time.time() - last_time > 3:
            last_time = time.time()
            has_face = False

    cv.ResetImageROI(image)

    cv.ShowImage('Faces', image)

    if cv.WaitKey(10) == 27:
        keep_running = False


def body(dev, ctx):
    global last_time

    if time.time() - last_time > 3 and not has_face:
        last_time = time.time()
        print 'NO FACE FOUND - Randomly tilting to see if I can find one..'
        led = random.randint(0, 6)
        freenect.set_led(dev, led)
        tilt = random.randint(0, 30)
        freenect.set_tilt_degs(dev, tilt)

    if not keep_running:
        raise freenect.Kill

if __name__ == "__main__":
    keep_running = True
    last_time = 0
    has_face = False
    # Build named Windows
    cv.NamedWindow('Faces')
    # Load Haar Cascades
    face_cascade = cv.Load('haar/frontalface.xml')
    eye_cascade = cv.Load('haar/eyes.xml')

    print('Press ESC in window to stop')
    freenect.runloop(
        video=find_faces,
        body=body
    )
