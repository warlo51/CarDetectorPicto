from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import io
from model import YOLOModel

app = Flask(__name__)
CORS(app)  # Ajoutez cette ligne pour activer CORS

model = YOLOModel()

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Ouvrir l'image
        image = Image.open(file.stream)

        # Effectuer la prédiction
        results = model.predict(image)

        # Annoter l'image
        annotated_image = model.annotate_image(image, results)

        # Convertir l'image annotée en un objet BytesIO pour l'envoyer dans la réponse
        img_io = io.BytesIO()
        annotated_image.save(img_io, format='JPEG')
        img_io.seek(0)

        # Envoyer l'image annotée avec le nom de fichier
        return send_file(
            img_io,
            mimetype='image/jpeg',
            download_name='annotated_image.jpg',
            as_attachment=True
        )

    except Exception as e:
        # Imprimer l'exception pour obtenir plus de détails
        print(f"Exception: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
