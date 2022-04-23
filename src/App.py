import json
from flask import Flask, jsonify, request
from io import BytesIO
import base64
from PIL import Image
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
from torch import no_grad
from flask_cors import CORS
import logging
feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-tiny-224")
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

app = Flask(__name__)
CORS(app)
PUBLIC = 'public'

@app.route(f'/{PUBLIC}/isAlive')
def is_alive():
    return 'Alive'


@app.route(f'/{PUBLIC}/predict', methods=['POST'])
def predict():
    image = request.json["image"]
    f = BytesIO()
    f.write(base64.b64decode(image))
    f.seek(0)
    image = Image.open(f).convert('RGB')

    if image is None:
        return jsonify(error="JSON content is empty"), 400

    inputs = feature_extractor(image, return_tensors="pt")
    with no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    output = {"class": model.config.id2label[predicted_label]}

    response = app.response_class(
        response=json.dumps(output),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.logger.info("App is starting ...")
    app.logger.info(
        f"Server is running, listening on port 8000")
    from waitress import serve

    serve(app, port=8080)