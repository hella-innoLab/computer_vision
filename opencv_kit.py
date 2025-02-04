import cv2
import numpy as np

# Funktion für Graustufen
def to_grayscale(image):
    # Überprüfe, ob das Bild bereits Graustufen hat
    if len(image.shape) == 3:
        # Stelle sicher, dass das Bild den richtigen Datentyp hat
        image = np.uint8(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image

# Funktion für Gaussian Blur (Tiefpassfilter)
def gaussian_blur(image, ksize=(5, 5), sigmaX=0):
    return cv2.GaussianBlur(image, ksize, sigmaX)

# Funktion für Laplacian Filter (Kantenverstärkung)
def laplacian_filter(image):
    return cv2.Laplacian(image, cv2.CV_64F)

# Funktion für Sobel Filter (Kantenverstärkung)
def sobel_filter(image):
    if len(image.shape) == 3:  # Falls Bild Farbbild ist, erst in Graustufen umwandeln
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)

# Funktion für Schärfefilter (Unsharp Mask)
def sharpen(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# Funktion für Hochpassfilter (Differenz von Original und Weichzeichnung)
def high_pass_filter(image):
    blurred = cv2.GaussianBlur(image, (21, 21), 0)
    return cv2.absdiff(image, blurred)

# Funktion für Tiefpassfilter (Weichzeichnung)
def low_pass_filter(image):
    return cv2.GaussianBlur(image, (21, 21), 0)

# Funktion für Thresholding (Schwellenwert)
def threshold_filter(image):
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresholded

# Funktion für Canny Edge Detection (Kantenfindung)
def canny_edge_detection(image, low_threshold=100, high_threshold=200):
    if len(image.shape) == 3:  # Falls Bild Farbbild ist, erst in Graustufen umwandeln
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray_image, low_threshold, high_threshold)
    else:
        image = np.uint8(image)
        return cv2.Canny(image, low_threshold, high_threshold)

    

# Funktion für Farbverstärkung (z.B. Sättigung erhöhen)
def color_enhance(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = hsv_image[..., 1] * 1.5  # Erhöhe die Sättigung
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Funktion für ein simples Rauschen (random noise)
def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.add(image, gauss)
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Funktion für den Hochpassfilter durch Frequenzbereich
def frequency_high_pass_filter(image):
    # Wenn das Bild ein Farbbild ist, konvertiere es in Graustufen
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Stelle sicher, dass das Bild den richtigen Datentyp hat (32-Bit Fließkommazahl)
    image_float = np.float32(image)  # Achte darauf, dass der Typ korrekt ist

    # Fourier-Transformation und Filterung im Frequenzbereich
    dft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Erstelle einen Lowpass-Filter und wende ihn auf das Bild an
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    dft_shift[crow-60:crow+60, ccol-60:ccol+60] = 0
    
    # Rücktransformation
    idft_shift = np.fft.ifftshift(dft_shift)
    idft = cv2.idft(idft_shift)
    img_back = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
    
    return np.uint8(img_back)

# Funktion, um mehrere Filter miteinander zu kombinieren
def apply_filters(image, filters):
    filtered_image = image
    for filter_func in filters:
        filtered_image = filter_func(filtered_image)
    return filtered_image

def find_contours(image_with_edges):
    # Konturen finden
    contours, _ = cv2.findContours(image_with_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ausgabe-Bild erstellen (schwarz)
    output = np.zeros_like(image_with_edges)  # Das gleiche Format wie das Eingangsbild, aber leer (schwarz)

    # Alle Konturen zeichnen
    cv2.drawContours(output, contours, -1, (255, 0, 0), 2)  # Alle Konturen zeichnen (blau)

    # Alle Konturen zu einer einzigen Form zusammenführen (Konvexe Hülle)
    all_contours = np.vstack(contours)  # Alle Konturen zusammenfügen

    # Begrenzendes Rechteck berechnen
    x, y, w, h = cv2.boundingRect(all_contours)  # Rechteck berechnen, das alle Konturen umschließt

    # Rechteck auf die Ausgabe zeichnen
    output_frame=frame.copy()
    cv2.rectangle(output, (x+1200, y+100), (x+1200 + w, y+100 + h), (0, 255, 0), 2)  # grünes Rechteck

    return output


def remove_noise_with_morphology(image_with_edges):
    kernel_size = 2  # Kernelgröße ändern
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Schwellenwert anwenden
    image_with_edges = cv2.adaptiveThreshold(image_with_edges, 255, 
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Erosion (kleines Rauschen entfernen) mit einer Iteration
    eroded_image = cv2.erode(image_with_edges, kernel, iterations=4)

    # Dilatation (benachbarte Konturen verbinden) mit einer Iteration
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=30)



    return dilated_image




def detect_and_draw_rectangles(image):

    # Konturen finden
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Umwandlung in Farbbild, damit Rechtecke sichtbar sind
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Rechtecke erkennen und zeichnen
    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)  # Genauigkeit verbessert
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Prüfen, ob die Kontur ein Viereck ist
        if len(approx) == 4:
            area = cv2.contourArea(contour)  # Berechne die Fläche
            if area>100:

                cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)  # Grün, Dicke 2
    
    return output



# Beispiel für die Verwendung der Filter
if __name__ == '__main__':

        image_path = r"C:\Users\VolggerF\Pictures\Camera Roll\volumenbestimmung_testframe.jpg"
        frame = cv2.imread(image_path)
        x, y, w, h = 1200, 100, 2000, 1300
        roi=frame[y:y+h,x:x+w]
        # Beispiel: Kombiniere mehrere Filter
        filters_to_apply_test1 = [
           
            to_grayscale,gaussian_blur,threshold_filter,canny_edge_detection,remove_noise_with_morphology,detect_and_draw_rectangles
            
            
        ]
        filters_to_apply=[to_grayscale]


        # Wende die Filter an
        processed_frame = apply_filters(roi, filters_to_apply)
        
        # Zeige das Ergebnis an

        max_display_size = 1200  # Maximal 800px in Breite oder Höhe
        orig_h, orig_w, _ = frame.shape
        scale_factor = min(max_display_size / orig_w, max_display_size / orig_h, 1.0)  # Maximalgröße limitieren

        new_width = int(processed_frame.shape[1] * scale_factor)
        new_height = int(processed_frame.shape[0] * scale_factor)

        processed_frame = cv2.resize(processed_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        frame=cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        roi=cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imshow('Gefiltertes Bild', processed_frame)
        cv2.imshow('orignal', frame)
        

        


        cv2.waitKey(0)
        cv2.destroyAllWindows()
