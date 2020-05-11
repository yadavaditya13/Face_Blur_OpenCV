# Face_Blur_OpenCV

This project is used to blur faces in image, liveCamera and videoFile.

The simple two steps that I followed to do this task were:
1) Face detection using pre-trained "res10_300x300_ssd_iter_140000.caffemodel".
2) Once got the ROIs pass them to GaussianBlur and get the blurred face as output and now you simply need to overLap this blurredFace with normal one in the frame / image.

There are more ways to blur (smoother) the face (image) other than GaussianBlur, they are MedianBlur, Blur, BilateralFilter as well.
