# model.py
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw

class YOLOModel:
    def __init__(self):
        # Charger le modèle pré-entraîné
        self.model = YOLO('newModel.pt')

    def predict(self, image):
        # Convertir l'image PIL en format OpenCV pour la détection
        open_cv_image = np.array(image.convert('RGB'))
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Effectuer la détection
        results = self.model(open_cv_image)

        # Retourner les résultats de la détection
        return results

    def annotate_image(self, image, results):
        # Convertir l'image PIL en format OpenCV pour ajouter des annotations
        open_cv_image = np.array(image.convert('RGB'))
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Dessiner les annotations
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()  # Ajouté pour obtenir les classes

            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box
                label = f'{self.model.names[int(cls)]} {conf:.2f}'  # Ajout du label

                # Dessiner le rectangle de détection
                cv2.rectangle(open_cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Dessiner la confiance et le label
                cv2.putText(open_cv_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convertir l'image annotée en format PIL pour la renvoyer
        annotated_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
        return annotated_image
