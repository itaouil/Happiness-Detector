"""
    The script does face detection,
    eye detection and on top of that
    smile detection using haar-like
    features.
"""

# Import OpenCV + sys
import cv2
import sys

# Load haar features
eye_cascade = cv2.CascadeClassifier('haar-features/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haar-features/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haar-features/haarcascade_smile.xml')

# Features detection
def detect(img_gray, img_colour):

    # Face detection
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    # Detect eyes and smile within ROI
    for (fx, fy, fw, fh) in faces:

        # Draw rectangles around faces
        cv2.rectangle(img_colour, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

        # Eye and smile detection Region of Interest (ROI)
        roi_gray   = img_gray[fy : fy + fh, fx : fx + fw]
        roi_colour = img_colour[fy : fy + fh, fx : fx + fw]

        # Eye detection + drawing
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 22)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangles around faces
            cv2.rectangle(roi_colour, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Smile detection + drawing
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            # Draw rectangles around faces
            cv2.rectangle(roi_colour, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    # Return image with drawn detections
    return img_colour

# Real time detection
def main(args):

    # Get video stream (internal camera)
    video = cv2.VideoCapture(0)

    while True:

        # Read the stream
        _, img_colour = video.read()

        # Convert image to grayscale
        img_gray = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)

        # Call function on images
        canvas = detect(img_gray, img_colour)

        # Show detections
        cv2.imshow("Detections", canvas)

        # Quit stream when q pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release stream and destroy cv2 windows
    # (aka release resources)
    video.release()
    cv2.destroyAllWindows()

# Execute the main
if __name__ == "__main__":
    main(sys.argv[0])
