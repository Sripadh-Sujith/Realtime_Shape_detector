import cv2
import numpy as np

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, use 1 for an external webcam

def getcontour(img, imgcontour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >200:  # Ignore small contours (noise)
            cv2.drawContours(imgcontour, cnt, -1, (255, 0, 0), 2)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            sides = len(approx)
            
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgcontour, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if sides == 3:
                obj = 'Triangle'
            elif sides == 4:
                aspect_ratio = w / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    obj = 'Square'
                else:
                    obj = 'Rectangle'
            else:
                obj = 'Circle'

            cv2.putText(imgcontour, obj, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

while True:
    success, frame = cap.read()  # Read frame from webcam
    if not success:
        break  # If frame not captured, exit

    imgcontour = frame.copy()
    imggray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgblur = cv2.GaussianBlur(imggray, (5, 5), 0)
    imgcanny = cv2.Canny(imgblur, 100, 200)

    getcontour(imgcanny, imgcontour)

    # Show results in real-time
    cv2.imshow('Contour Detection', imgcontour)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
