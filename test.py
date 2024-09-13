from ultralytics import YOLO

# Load a model
#model = YOLO("newModel.pt")  # pretrained YOLOv8n model

#results = model.val(data="carIcon.yaml", conf=0.5, imgsz=640)

# Charger le modèle
model = YOLO("newModel.pt")  # Charger votre modèle YOLOv8 personnalisé

# Tester une image spécifique
results = model.predict(source=["t3.jpeg"], conf=0.9, imgsz=640)

# Afficher les résultats (facultatif)
for result in results:
    result.show()
