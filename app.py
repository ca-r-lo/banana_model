from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = "banana_model.tflite"

CLASS_NAMES = [
    "Black Sigatoka",
    "BBMV",
    "Healthy",
    "Banana Insect Pest",
    "Moko Disease",
    "Fusarium Wilt",
    "Yellow Sigatoka"
]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32)

    # Keep this if your model was trained using 0–1 normalized images.
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def run_prediction(image_bytes):
    input_data = preprocess_image(image_bytes)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted_index = int(np.argmax(output_data))
    confidence = float(output_data[predicted_index])

    scores = [
        {
            "label": CLASS_NAMES[i],
            "score": float(output_data[i])
        }
        for i in range(len(CLASS_NAMES))
    ]

    return {
        "prediction_index": predicted_index,
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "scores": scores
    }


@app.route("/")
def home():
    return render_template("index.html", classes=CLASS_NAMES)


@app.route("/test")
def test_page():
    return render_template("test.html", classes=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    result = run_prediction(image_bytes)

    return jsonify(result)


@app.route("/bulk-test", methods=["POST"])
def bulk_test():
    expected_label = request.form.get("expected_label")
    files = request.files.getlist("images")

    if not expected_label:
        return jsonify({"error": "Expected label is required"}), 400

    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    total = 0
    correct = 0
    results = []

    class_summary = {
        class_name: {
            "predicted_count": 0
        }
        for class_name in CLASS_NAMES
    }

    for file in files:
        image_bytes = file.read()
        prediction = run_prediction(image_bytes)

        is_correct = prediction["prediction"] == expected_label

        total += 1
        if is_correct:
            correct += 1

        class_summary[prediction["prediction"]]["predicted_count"] += 1

        results.append({
            "filename": file.filename,
            "expected": expected_label,
            "prediction": prediction["prediction"],
            "confidence": prediction["confidence"],
            "correct": is_correct,
            "scores": prediction["scores"]
        })

    accuracy = correct / total if total > 0 else 0

    return jsonify({
        "expected_label": expected_label,
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": accuracy,
        "class_summary": class_summary,
        "results": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)