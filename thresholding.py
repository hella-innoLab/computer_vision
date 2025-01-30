import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(2)  # Webcam

while True:
    ret, frame= cap.read()
    if not ret:
        break

    roi = frame[80:300, 100:500]

    img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    img = cv2.medianBlur(img, 5)
    
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2)
    
    # Finde Konturen und zeichne Rechtecke
    def draw_rectangles(image):
        edges = cv2.Canny(image, 50, 150)

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_2=frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 200:  # Ignoriere kleine Veränderungen (Schwellenwert)
                (x, y, w, h) = cv2.boundingRect(contour)
                # Offset-Korrektur (ROI beginnt bei x=100, y=80)
                
                cv2.rectangle(frame_2, (x + 100, y + 80), (x + w + 100, y + h + 80), (0, 0, 255), 2)
        return frame_2

    # Zeichne Rechtecke auf die Schwellenbilder
    th1_with_rect = draw_rectangles(th1.copy())
    th2_with_rect = draw_rectangles(th2.copy())
    th3_with_rect = draw_rectangles(th3.copy())

    # Setze die Titel und Bilder für die Darstellung
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1_with_rect, th2_with_rect, th3_with_rect]
    
    cv2.imshow('roi',roi)
    cv2.imshow('Test0', img)
    cv2.imshow('Test1', th1_with_rect)
    cv2.imshow('Test2', th2_with_rect)
    cv2.imshow('Test3', th3_with_rect)



    '''for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()'''

    # Beende das Programm mit der 'q'-Taste
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
