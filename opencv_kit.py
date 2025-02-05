import cv2
from sklearn.cluster import DBSCAN
from pyzbar.pyzbar import decode
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


# Funktion für Thresholding (Schwellenwert)
def threshold_filter(image):
    image = np.uint8(image)
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresholded

# Funktion für Canny Edge Detection (Kantenfindung)
def canny_edge_detection(image, low_threshold=100, high_threshold=200):
    if len(image.shape) == 3:  # Falls Bild Farbbild ist, erst in Graustufen umwandeln
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray_image, low_threshold, high_threshold)
    else:
        image = np.uint8(image)
        return cv2.Canny(image, low_threshold, high_threshold)

    
def detect_rectangle_statistical(image):
    # Alle weißen Punkte finden
    white_points = np.column_stack(np.where(image == 255))  # (y, x)

    if len(white_points) == 0:
        return image

    # 1. DBSCAN für Rauschfilterung**
    clustering = DBSCAN(eps=10, min_samples=10).fit(white_points)  # eps=5: Max. Abstand
    labels = clustering.labels_

    # 2. Finden des flächenmäßig größten Clusters
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Das Bild in BGR umwandeln, um Rechtecke zu zeichnen
    output_image=roi.copy()
    
    area=0
    
    for label in set(labels):
        if label == -1:  # Cluster -1 ist für Rauschen, ignorieren
            continue
        
        # Punkte für das aktuelle Cluster filtern
        cluster_points = white_points[labels == label]

        if len(cluster_points) < 100:  # Wenn das Cluster zu klein ist, überspringen
            continue

        # Axis-Aligned Bounding Box für das Cluster berechnen
        min_x = np.min(cluster_points[:, 1])
        max_x = np.max(cluster_points[:, 1])
        min_y = np.min(cluster_points[:, 0])
        max_y = np.max(cluster_points[:, 0])

        temp_area=(max_x-min_x)*(max_y-min_y)
        if temp_area>area:
            area=temp_area
            fin_min_x=min_x
            fin_min_y=min_y
            fin_max_x=max_x
            fin_max_y=max_y
            
        if (max_x-min_x)*(max_y-min_y)>=800:
            cv2.rectangle(output_image, (fin_min_x, fin_min_y), (fin_max_x, fin_max_y), (0, 255, 0), 2)
                # Berechne die Länge des Rechtecks
            factor=0.35450517
            length = round((fin_max_x - fin_min_x)*factor,0)
            width = round((fin_max_y - fin_min_y)*factor,0)
            
            # Beschrifte das Rechteck mit seiner Länge und Breite
            text_length = f'Lange: {length} mm'
            text_width = f'Breite: {width} mm'
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            color = (0, 0, 255)  # Rot
            thickness = 4
            
            # Textpositionen
            position_length = (fin_min_x, fin_min_y - 20)
            position_width = (fin_min_x, fin_min_y - 80)
            
            # Füge die Texte zur Ausgabe hinzu
            cv2.putText(output_image, text_length, position_length, font, font_scale, color, thickness)
            cv2.putText(output_image, text_width, position_width, font, font_scale, color, thickness)
    
    return output_image


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


# Funktion, um mehrere Filter miteinander zu kombinieren
def apply_filters(image, filters):
    filtered_image = image
    for filter_func in filters:
        filtered_image = filter_func(filtered_image)
    return filtered_image



def remove_noise_with_morphology(image_with_edges):
    kernel_size = 2  # Kernelgröße ändern
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Erosion (kleines Rauschen entfernen) mit einer Iteration
    eroded_image = cv2.erode(image_with_edges, kernel, iterations=2)
    # Dilatation (benachbarte Konturen verbinden) mit einer Iteration
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=3)
    
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

def invert_image(image):
    return cv2.bitwise_not(image)


    
def BarcodeReader(img): 
      
    # Decode the barcode image 
    detectedBarcodes = decode(img) 
       
    # If not detected then print the message 
    if not detectedBarcodes: 
        print("Barcode Not Detected or your barcode is blank/corrupted!") 
    else: 
        
          # Traverse through all the detected barcodes in image 
        for barcode in detectedBarcodes:   
            
            # Locate the barcode position in image 
            (x, y, w, h) = barcode.rect 
              
            # Put the rectangle in image using  
            # cv2 to highlight the barcode 
            cv2.rectangle(img, (x-10, y-10), 
                          (x + w+10, y + h+10),  
                          (255, 0, 0), 2) 
              
            if barcode.data!="": 
                
            # Print the barcode data 
                print(barcode.data) 
                print(barcode.type) 
                  
    #Display the image 
    cv2.imshow("Image", img) 




# Beispiel für die Verwendung der Filter
if __name__ == '__main__':

        image_path = r"C:\Users\VolggerF\Pictures\Camera Roll\WIN_20250204_14_59_45_Pro.jpg"
        frame = cv2.imread(image_path)
        x, y, w, h = 200, 500, 3000, 2500
        roi=frame[y:y+h,x:x+w]
        # Beispiel: Kombiniere mehrere Filter
        filters_to_apply_approach1 = [
           
            to_grayscale,gaussian_blur,threshold_filter,canny_edge_detection,detect_and_draw_rectangles
            
            
        ]
        filters_to_apply_approach2=[to_grayscale,gaussian_blur,threshold_filter,remove_noise_with_morphology,detect_rectangle_statistical]

        filters_to_apply=[to_grayscale,gaussian_blur,threshold_filter,remove_noise_with_morphology,detect_rectangle_statistical


        # Wende die Filter an
        processed_frame = apply_filters(roi, filters_to_apply)
        
        # Zeige das Ergebnis an

        max_display_size = 1500  # Maximal 800px in Breite oder Höhe
        orig_h, orig_w, _ = frame.shape
        scale_factor = min(max_display_size / orig_w, max_display_size / orig_h, 1.0)  # Maximalgröße limitieren

        new_width = int(processed_frame.shape[1] * scale_factor)
        new_height = int(processed_frame.shape[0] * scale_factor)

        processed_frame = cv2.resize(processed_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        frame=cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        roi=cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_AREA)









        filters_to_apply_barcode=[to_grayscale]
        processed_barcode_frame=apply_filters(frame,filters_to_apply_barcode)

        #BarcodeReader(processed_barcode_frame)





        cv2.imshow('Gefiltertes Bild', processed_frame)
        cv2.imshow('original', frame)
        

        


        cv2.waitKey(0)
        cv2.destroyAllWindows()
