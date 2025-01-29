import cv2
import numpy as np

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
    ret, frame = cap.read()
    if not ret:
        break

    # Wandelt das aktuelle Frame in Graustufen um
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Berechne die absolute Differenz zwischen dem aktuellen Frame und dem Referenz-Frame
    frame_diff = cv2.absdiff(reference_gray, gray_frame)

    # Wende eine Schwellenwertoperation an, um nur signifikante Änderungen zu sehen
    _, thresholded = cv2.threshold(frame_diff, 60, 255, cv2.THRESH_BINARY)

    # Finde die Konturen der Veränderungen
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Zeichne die gefundenen Veränderungen auf das Bild
    for contour in contours:
        if cv2.contourArea(contour) > 400:  # Ignoriere kleine Veränderungen (Schwellenwert)
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Zeige das Ergebnis an (Frame mit markierten Änderungen)
    cv2.imshow('Changes Marked', frame)
    
    # Zeige das Graustufenbild
    cv2.imshow('Gray Frame', gray_frame)
    
    # Zeige die absolute Differenz
    cv2.imshow('Frame Difference', frame_diff)
    
    # Zeige das Schwellenwertbild (signifikante Änderungen)
    cv2.imshow('Thresholded', thresholded)

    # Beende das Programm mit der 'q'-Taste
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
