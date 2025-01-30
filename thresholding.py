import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt


cap = cv2.VideoCapture(2)  # Webcam

# Warten auf den ersten Frame und als Referenz speichern
ret, reference_frame = cap.read()
if not ret:
    print("Fehler beim Aufnehmen des ersten Frames!")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Wandelt den ersten Frame in Graustufen um (Referenz)
reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

# Hole die Größe des Referenzbildes
h, w = reference_gray.shape[:2]

while True:
    ret, img = cap.read()
    if not ret:
        break

    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    img = cv2.medianBlur(img,5)
    
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    
    titles = ['Original Image', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    # Beende das Programm mit der 'q'-Taste
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
