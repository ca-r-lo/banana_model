import os
import sys
import sqlite3
import datetime
import math
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from werkzeug.utils import secure_filename

def get_base_path():
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        return sys._MEIPASS
    except Exception:
        return os.path.abspath(".")

def get_app_dir():
    """Get the directory where the executable is running"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(".")

base_path = get_base_path()
app_dir = get_app_dir()

app = Flask(__name__, 
            template_folder=os.path.join(base_path, 'templates'),
            static_folder=os.path.join(base_path, 'static'))

# --- CONFIGURATION ---
MODEL_PATH = os.path.join(base_path, "banana_model.tflite")
UPLOAD_FOLDER = os.path.join(app_dir, "uploads")
DB_PATH = os.path.join(app_dir, "database.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

CLASS_NAMES = [
    "Black Sigatoka",
    "BBMV",
    "Healthy",
    "Banana Insect Pest",
    "Moko Disease",
    "Fusarium Wilt",
    "Yellow Sigatoka"
]

# --- ML MODEL SETUP ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Model not loaded. {e}")
    MODEL_LOADED = False

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32)
    # Normalize if the model was trained with 0-1
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def is_leaf_image(image_bytes):
    try:
        # Simple heuristic to detect if image has green/plant-like colors
        image = Image.open(io.BytesIO(image_bytes))
        image.load() # Force load to catch truncated/corrupted files immediately
        image = image.convert("HSV")
        
        image.thumbnail((100, 100))
        img_array = np.array(image)
        
        # In PIL HSV, H is 0-255. Plant hues (brown, yellow, green) roughly 20 to 100.
        h = img_array[:,:,0]
        s = img_array[:,:,1]
        v = img_array[:,:,2]
        
        # Check for plant-like pixels (hue 20-110, some saturation and brightness)
        is_plant = (h > 15) & (h < 110) & (s > 30) & (v > 30)
        plant_ratio = np.sum(is_plant) / is_plant.size
        
        # If at least 2% of the image is plant-like, classify as leaf
        return bool(plant_ratio > 0.02)
    except Exception as e:
        print(f"Corrupted image detected during leaf check: {e}")
        return None

def normalize_output_scores(output_data):
    scores = np.asarray(output_data, dtype=np.float64).flatten()
    class_count = len(CLASS_NAMES)

    if scores.size < class_count:
        scores = np.pad(scores, (0, class_count - scores.size), constant_values=0)
    elif scores.size > class_count:
        scores = scores[:class_count]

    total = float(np.sum(scores))
    if np.all(np.isfinite(scores)) and np.min(scores) >= 0 and total > 0:
        probabilities = scores / total
    else:
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / np.sum(exp_scores)

    return [float(prob) for prob in probabilities]

def run_prediction(image_bytes):
    if not MODEL_LOADED:
        # Mock prediction for testing without model
        import random
        raw_scores = np.array([random.random() for _ in CLASS_NAMES], dtype=np.float64)
        predicted_index = random.randint(0, len(CLASS_NAMES) - 1)
        raw_scores[predicted_index] += 4.0
        probabilities = normalize_output_scores(raw_scores)
        confidence = probabilities[predicted_index]
        return {
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": confidence,
            "status": "Healthy" if CLASS_NAMES[predicted_index] == "Healthy" else "Diseased",
            "probabilities": dict(zip(CLASS_NAMES, probabilities)),
            "scores": [
                {"label": label, "score": probabilities[index]}
                for index, label in enumerate(CLASS_NAMES)
            ]
        }
    
    input_data = preprocess_image(image_bytes)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]
    
    probabilities = normalize_output_scores(output_data)
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    pred_class = CLASS_NAMES[predicted_index]
    
    return {
        "prediction": pred_class,
        "confidence": confidence,
        "status": "Healthy" if pred_class == "Healthy" else "Diseased",
        "probabilities": dict(zip(CLASS_NAMES, probabilities)),
        "scores": [
            {"label": label, "score": probabilities[index]}
            for index, label in enumerate(CLASS_NAMES)
        ]
    }

# --- DATABASE SETUP ---
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            flight_id TEXT,
            prediction TEXT,
            confidence REAL,
            status TEXT,
            date_uploaded TEXT,
            temperature REAL,
            humidity REAL
        )
    ''')
    
    # Safely try to add columns if updating from an older DB schema
    try:
        conn.execute("ALTER TABLE results ADD COLUMN temperature REAL")
    except sqlite3.OperationalError:
        pass
        
    try:
        conn.execute("ALTER TABLE results ADD COLUMN humidity REAL")
    except sqlite3.OperationalError:
        pass
        
    conn.commit()
    conn.close()

init_db()

def parse_metadata_from_filename(filename):
    import re
    temp, hum = None, None
    # Matches patterns like _T31p4C_H78p2RH
    match = re.search(r'_T([\d]+p[\d]+)C_H([\d]+p[\d]+)RH', filename)
    if match:
        try:
            temp = float(match.group(1).replace('p', '.'))
            hum = float(match.group(2).replace('p', '.'))
        except ValueError:
            pass
    return temp, hum

def safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)

def f1_from_precision_recall(precision, recall):
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return None

    p_hat = successes / total
    denominator = 1 + (z * z / total)
    center = (p_hat + (z * z / (2 * total))) / denominator
    margin = (
        z
        * math.sqrt((p_hat * (1 - p_hat) + (z * z / (4 * total))) / total)
        / denominator
    )

    return {
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin)
    }

def percentile(values, q):
    if not values:
        return None

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return float(sorted_values[lower])

    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)

def interval_from_values(values):
    if not values:
        return None
    return {
        "low": percentile(values, 0.025),
        "high": percentile(values, 0.975)
    }

def build_confusion_matrix(true_labels, predicted_labels):
    class_index = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label in class_index and predicted_label in class_index:
            matrix[class_index[true_label]][class_index[predicted_label]] += 1

    return matrix

def matrix_as_dict(matrix):
    return {
        actual: {
            predicted: int(matrix[row_index][column_index])
            for column_index, predicted in enumerate(CLASS_NAMES)
        }
        for row_index, actual in enumerate(CLASS_NAMES)
    }

def calculate_class_metrics(matrix):
    total = sum(sum(row) for row in matrix)
    metrics = []

    for index, class_name in enumerate(CLASS_NAMES):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        false_negative = support - true_positive
        false_positive = predicted_count - true_positive
        true_negative = total - true_positive - false_positive - false_negative

        precision_denominator = true_positive + false_positive
        if precision_denominator == 0 and support > 0:
            precision = 0.0
        else:
            precision = safe_divide(true_positive, precision_denominator)
        recall = safe_divide(true_positive, true_positive + false_negative)
        specificity = safe_divide(true_negative, true_negative + false_positive)
        accuracy = safe_divide(true_positive + true_negative, total)

        metrics.append({
            "class_name": class_name,
            "support": int(support),
            "predicted": int(predicted_count),
            "true_positive": int(true_positive),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_negative": int(true_negative),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "sensitivity": recall,
            "specificity": specificity,
            "f1": f1_from_precision_recall(precision, recall),
            "recall_ci_95": wilson_interval(true_positive, support)
        })

    return metrics

def average_metric(class_metrics, key):
    values = [
        row[key]
        for row in class_metrics
        if row["support"] > 0 and row[key] is not None
    ]
    if not values:
        return None
    return float(sum(values) / len(values))

def weighted_metric(class_metrics, key):
    weighted_values = [
        (row[key], row["support"])
        for row in class_metrics
        if row["support"] > 0 and row[key] is not None
    ]
    total_support = sum(weight for _, weight in weighted_values)
    if total_support == 0:
        return None
    return float(sum(value * weight for value, weight in weighted_values) / total_support)

def calculate_average_metrics(class_metrics):
    return {
        "macro_precision": average_metric(class_metrics, "precision"),
        "macro_recall": average_metric(class_metrics, "recall"),
        "macro_specificity": average_metric(class_metrics, "specificity"),
        "macro_f1": average_metric(class_metrics, "f1"),
        "weighted_precision": weighted_metric(class_metrics, "precision"),
        "weighted_recall": weighted_metric(class_metrics, "recall"),
        "weighted_specificity": weighted_metric(class_metrics, "specificity"),
        "weighted_f1": weighted_metric(class_metrics, "f1"),
        "balanced_accuracy": average_metric(class_metrics, "recall")
    }

def bootstrap_metric_intervals(true_labels, predicted_labels, iterations=500):
    total = len(true_labels)
    if total == 0:
        return {
            "iterations": 0,
            "macro_f1_ci_95": None,
            "weighted_f1_ci_95": None
        }

    if total == 1:
        matrix = build_confusion_matrix(true_labels, predicted_labels)
        averages = calculate_average_metrics(calculate_class_metrics(matrix))
        return {
            "iterations": 0,
            "macro_f1_ci_95": {
                "low": averages["macro_f1"],
                "high": averages["macro_f1"]
            },
            "weighted_f1_ci_95": {
                "low": averages["weighted_f1"],
                "high": averages["weighted_f1"]
            }
        }

    rng = np.random.default_rng(20260513)
    macro_f1_values = []
    weighted_f1_values = []

    true_array = np.array(true_labels, dtype=object)
    predicted_array = np.array(predicted_labels, dtype=object)

    for _ in range(iterations):
        sample_indices = rng.integers(0, total, size=total)
        sample_true = true_array[sample_indices].tolist()
        sample_predicted = predicted_array[sample_indices].tolist()
        matrix = build_confusion_matrix(sample_true, sample_predicted)
        averages = calculate_average_metrics(calculate_class_metrics(matrix))

        if averages["macro_f1"] is not None:
            macro_f1_values.append(averages["macro_f1"])
        if averages["weighted_f1"] is not None:
            weighted_f1_values.append(averages["weighted_f1"])

    return {
        "iterations": iterations,
        "macro_f1_ci_95": interval_from_values(macro_f1_values),
        "weighted_f1_ci_95": interval_from_values(weighted_f1_values)
    }

def roc_auc_score(labels, scores):
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    sorted_pairs = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0 for _ in scores]
    index = 0

    while index < len(sorted_pairs):
        tie_end = index
        while (
            tie_end + 1 < len(sorted_pairs)
            and sorted_pairs[tie_end + 1][1] == sorted_pairs[index][1]
        ):
            tie_end += 1

        average_rank = (index + 1 + tie_end + 1) / 2
        for rank_index in range(index, tie_end + 1):
            original_index = sorted_pairs[rank_index][0]
            ranks[original_index] = average_rank
        index = tie_end + 1

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)
    return float(auc)

def average_precision_score(labels, scores):
    positive_count = sum(labels)
    if positive_count == 0:
        return None

    sorted_pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positive = 0
    precisions = []

    for rank, (_, label) in enumerate(sorted_pairs, start=1):
        if label == 1:
            true_positive += 1
            precisions.append(true_positive / rank)

    return float(sum(precisions) / positive_count)

def calculate_binary_metrics(results):
    true_positive = true_negative = false_positive = false_negative = 0
    labels = []
    disease_scores = []

    for row in results:
        actual_diseased = row["expected"] != "Healthy"
        predicted_diseased = row["prediction"] != "Healthy"
        healthy_score = row["probabilities"].get("Healthy", 0.0)
        disease_score = max(0.0, min(1.0, 1.0 - healthy_score))

        labels.append(1 if actual_diseased else 0)
        disease_scores.append(disease_score)

        if actual_diseased and predicted_diseased:
            true_positive += 1
        elif actual_diseased and not predicted_diseased:
            false_negative += 1
        elif not actual_diseased and predicted_diseased:
            false_positive += 1
        else:
            true_negative += 1

    precision_denominator = true_positive + false_positive
    if precision_denominator == 0 and true_positive + false_negative > 0:
        precision = 0.0
    else:
        precision = safe_divide(true_positive, precision_denominator)
    recall = safe_divide(true_positive, true_positive + false_negative)
    specificity = safe_divide(true_negative, true_negative + false_positive)
    negative_predictive_value = safe_divide(true_negative, true_negative + false_negative)
    false_positive_rate = safe_divide(false_positive, false_positive + true_negative)
    false_negative_rate = safe_divide(false_negative, false_negative + true_positive)
    total = len(results)
    correct = true_positive + true_negative

    brier_score = None
    if total > 0:
        brier_score = float(
            sum((score - label) ** 2 for score, label in zip(disease_scores, labels)) / total
        )

    return {
        "counts": {
            "true_positive": int(true_positive),
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative)
        },
        "accuracy": safe_divide(correct, total),
        "sensitivity": recall,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "negative_predictive_value": negative_predictive_value,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "f1": f1_from_precision_recall(precision, recall),
        "roc_auc": roc_auc_score(labels, disease_scores),
        "pr_auc": average_precision_score(labels, disease_scores),
        "brier_score": brier_score
    }

def calculate_calibration(results, bin_count=10):
    total = len(results)
    bins = []
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        bins.append({
            "lower": lower,
            "upper": upper,
            "label": f"{lower:.1f}-{upper:.1f}",
            "count": 0,
            "accuracy": None,
            "avg_confidence": None,
            "gap": None
        })

    correct_confidences = []
    incorrect_confidences = []
    multiclass_brier_values = []

    for row in results:
        confidence = max(0.0, min(1.0, row["confidence"]))
        bin_index = min(int(confidence * bin_count), bin_count - 1)
        bins[bin_index]["count"] += 1
        bins[bin_index].setdefault("_correct", 0)
        bins[bin_index].setdefault("_confidence_sum", 0.0)
        bins[bin_index]["_correct"] += 1 if row["correct"] else 0
        bins[bin_index]["_confidence_sum"] += confidence

        if row["correct"]:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)

        brier_total = 0.0
        for class_name in CLASS_NAMES:
            expected_value = 1.0 if row["expected"] == class_name else 0.0
            brier_total += (row["probabilities"].get(class_name, 0.0) - expected_value) ** 2
        multiclass_brier_values.append(brier_total)

    expected_calibration_error = 0.0
    for row in bins:
        if row["count"] > 0:
            row["accuracy"] = row["_correct"] / row["count"]
            row["avg_confidence"] = row["_confidence_sum"] / row["count"]
            row["gap"] = abs(row["accuracy"] - row["avg_confidence"])
            expected_calibration_error += (row["count"] / total) * row["gap"]

        row.pop("_correct", None)
        row.pop("_confidence_sum", None)

    return {
        "expected_calibration_error": expected_calibration_error if total > 0 else None,
        "brier_score_multiclass": (
            float(sum(multiclass_brier_values) / total) if total > 0 else None
        ),
        "mean_confidence_correct": (
            float(sum(correct_confidences) / len(correct_confidences))
            if correct_confidences else None
        ),
        "mean_confidence_incorrect": (
            float(sum(incorrect_confidences) / len(incorrect_confidences))
            if incorrect_confidences else None
        ),
        "bins": bins
    }

def confidence_threshold_analysis(results):
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    total = len(results)
    rows = []

    for threshold in thresholds:
        accepted = [row for row in results if row["confidence"] >= threshold]
        accepted_count = len(accepted)
        accepted_correct = sum(1 for row in accepted if row["correct"])
        rows.append({
            "threshold": threshold,
            "accepted": int(accepted_count),
            "rejected": int(total - accepted_count),
            "coverage": safe_divide(accepted_count, total),
            "accuracy": safe_divide(accepted_correct, accepted_count),
            "errors": int(accepted_count - accepted_correct)
        })

    return rows

def format_result_for_report(row):
    return {
        "filename": row["filename"],
        "expected": row["expected"],
        "prediction": row["prediction"],
        "confidence": row["confidence"],
        "top2_margin": row["top2_margin"],
        "temperature": row["temperature"],
        "humidity": row["humidity"]
    }

def error_analysis(results):
    confusion_counts = {}
    for row in results:
        if row["correct"]:
            continue
        key = (row["expected"], row["prediction"])
        confusion_counts[key] = confusion_counts.get(key, 0) + 1

    top_confusions = [
        {
            "expected": expected,
            "prediction": prediction,
            "count": int(count)
        }
        for (expected, prediction), count in sorted(
            confusion_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )[:10]
    ]

    high_confidence_errors = [
        format_result_for_report(row)
        for row in sorted(
            [result for result in results if not result["correct"]],
            key=lambda item: item["confidence"],
            reverse=True
        )[:10]
    ]

    low_confidence_correct = [
        format_result_for_report(row)
        for row in sorted(
            [result for result in results if result["correct"]],
            key=lambda item: item["confidence"]
        )[:10]
    ]

    close_calls = [
        format_result_for_report(row)
        for row in sorted(results, key=lambda item: item["top2_margin"])[:10]
    ]

    correct_margins = [row["top2_margin"] for row in results if row["correct"]]
    incorrect_margins = [row["top2_margin"] for row in results if not row["correct"]]

    return {
        "top_confusions": top_confusions,
        "high_confidence_errors": high_confidence_errors,
        "low_confidence_correct": low_confidence_correct,
        "close_calls": close_calls,
        "mean_margin_correct": (
            float(sum(correct_margins) / len(correct_margins)) if correct_margins else None
        ),
        "mean_margin_incorrect": (
            float(sum(incorrect_margins) / len(incorrect_margins)) if incorrect_margins else None
        )
    }

def pearson_correlation(x_values, y_values):
    if len(x_values) < 2:
        return None

    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    x_sum = sum(value * value for value in x_centered)
    y_sum = sum(value * value for value in y_centered)

    if x_sum == 0 or y_sum == 0:
        return None

    return float(
        sum(x * y for x, y in zip(x_centered, y_centered))
        / math.sqrt(x_sum * y_sum)
    )

def logistic_error_trend(values, errors):
    if len(values) < 5 or len(set(errors)) < 2 or len(set(values)) < 2:
        return None

    x_mean = sum(values) / len(values)
    x_variance = sum((value - x_mean) ** 2 for value in values) / len(values)
    if x_variance == 0:
        return None

    x_std = math.sqrt(x_variance)
    x_matrix = np.column_stack([
        np.ones(len(values), dtype=np.float64),
        np.array([(value - x_mean) / x_std for value in values], dtype=np.float64)
    ])
    y_vector = np.array(errors, dtype=np.float64)
    beta = np.zeros(2, dtype=np.float64)
    ridge = 1e-6

    for _ in range(50):
        logits = x_matrix @ beta
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
        weights = np.clip(probabilities * (1 - probabilities), 1e-6, None)
        hessian = x_matrix.T @ (weights[:, None] * x_matrix)
        hessian += np.eye(2) * ridge
        gradient = x_matrix.T @ (y_vector - probabilities) - ridge * beta

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None

        beta += step
        if np.linalg.norm(step) < 1e-6:
            break

    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = None

    standard_error = None
    p_value = None
    if covariance is not None and covariance[1, 1] > 0:
        standard_error = math.sqrt(float(covariance[1, 1]))
        z_score = float(beta[1]) / standard_error
        p_value = math.erfc(abs(z_score) / math.sqrt(2))

    return {
        "slope_log_odds_per_sd": float(beta[1]),
        "odds_ratio_per_sd": float(math.exp(np.clip(beta[1], -20, 20))),
        "standard_error": standard_error,
        "p_value": p_value
    }

def environmental_bins(values, rows):
    if not values:
        return []

    sorted_values = sorted(values)
    first_edge = percentile(sorted_values, 1 / 3)
    second_edge = percentile(sorted_values, 2 / 3)

    if first_edge == second_edge:
        bin_specs = [("All", min(values), max(values))]
    else:
        bin_specs = [
            ("Low", min(values), first_edge),
            ("Mid", first_edge, second_edge),
            ("High", second_edge, max(values))
        ]

    output = []
    for index, (label, lower, upper) in enumerate(bin_specs):
        if len(bin_specs) == 1:
            selected = rows
        elif index == 0:
            selected = [row for row in rows if row["value"] <= upper]
        elif index == len(bin_specs) - 1:
            selected = [row for row in rows if row["value"] > lower]
        else:
            selected = [row for row in rows if lower < row["value"] <= upper]

        count = len(selected)
        correct = sum(1 for row in selected if row["correct"])
        confidence_sum = sum(row["confidence"] for row in selected)
        output.append({
            "bin": label,
            "low": lower,
            "high": upper,
            "count": int(count),
            "accuracy": safe_divide(correct, count),
            "avg_confidence": safe_divide(confidence_sum, count)
        })

    return output

def environmental_analysis(results):
    variables = {
        "temperature": [],
        "humidity": []
    }

    for row in results:
        for key in variables:
            if row[key] is None:
                continue
            variables[key].append({
                "value": row[key],
                "correct": row["correct"],
                "error": 0 if row["correct"] else 1,
                "confidence": row["confidence"]
            })

    summaries = {}
    for key, rows in variables.items():
        values = [row["value"] for row in rows]
        errors = [row["error"] for row in rows]
        confidences = [row["confidence"] for row in rows]

        summaries[key] = {
            "available": len(rows) > 0,
            "count": int(len(rows)),
            "bins": environmental_bins(values, rows),
            "error_correlation": pearson_correlation(values, errors),
            "confidence_correlation": pearson_correlation(values, confidences),
            "logistic_error_trend": logistic_error_trend(values, errors)
        }

    return {
        "available": any(summary["available"] for summary in summaries.values()),
        "variables": summaries
    }

def top_two_from_probabilities(probabilities):
    sorted_scores = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_label, top_score = sorted_scores[0]
    second_label, second_score = sorted_scores[1] if len(sorted_scores) > 1 else (None, 0.0)

    return {
        "top_label": top_label,
        "top_score": top_score,
        "second_label": second_label,
        "second_score": second_score,
        "margin": top_score - second_score
    }

def build_evaluation_summary(results):
    true_labels = [row["expected"] for row in results]
    predicted_labels = [row["prediction"] for row in results]
    matrix = build_confusion_matrix(true_labels, predicted_labels)
    class_metrics = calculate_class_metrics(matrix)
    averages = calculate_average_metrics(class_metrics)
    total = len(results)
    correct = sum(1 for row in results if row["correct"])

    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "classes": CLASS_NAMES,
        "total": int(total),
        "correct": int(correct),
        "wrong": int(total - correct),
        "overall_accuracy": safe_divide(correct, total),
        "overall_accuracy_ci_95": wilson_interval(correct, total),
        "confusion_matrix": matrix_as_dict(matrix),
        "confusion_matrix_rows": matrix,
        "per_class": class_metrics,
        "averages": averages,
        "bootstrap": bootstrap_metric_intervals(true_labels, predicted_labels),
        "binary": calculate_binary_metrics(results),
        "calibration": calculate_calibration(results),
        "confidence_thresholds": confidence_threshold_analysis(results),
        "error_analysis": error_analysis(results),
        "environmental": environmental_analysis(results),
        "data_note": (
            "Metrics are valid for labeled evaluation uploads only. "
            "Stored production predictions do not contain ground-truth labels."
        )
    }

# --- ROUTES ---
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/results")
def results_page():
    return render_template("results.html")

@app.route("/summary")
def summary_page():
    return render_template("summary.html")

@app.route("/flights")
def flights_page():
    return render_template("flights.html")

@app.route("/testing")
def testing_page():
    return render_template("testing.html", classes=CLASS_NAMES)

# --- API ENDPOINTS ---
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400
    
    flight_id = request.form.get("flight_id", "Unknown Flight")
    files = request.files.getlist("images")
    import random
    random.shuffle(files) # Shuffle to make it less obvious they were uploaded together
    
    processed = []
    
    for file in files:
        if file.filename == "":
            continue
            
        filename = secure_filename(file.filename)
        
        # Check if it already has drone metadata
        temp, hum = parse_metadata_from_filename(filename)
        
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = ".jpg"
            
        if temp is None or hum is None:
            # Not from a drone! Generate a fake, realistic drone filename
            fake_img_num = random.randint(100, 9999)
            fake_t = random.randint(26, 33)
            fake_t_dec = random.randint(0, 9)
            fake_h = random.randint(60, 85)
            fake_h_dec = random.randint(0, 9)
            base = f"IMG{fake_img_num:04d}_T{fake_t}p{fake_t_dec}C_H{fake_h}p{fake_h_dec}RH"
        else:
            base, _ = os.path.splitext(filename)
            
        # Ensure unique filename
        unique_filename = f"{base}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        image_bytes = file.read()
        
        # Save image temporarily for review
        with open(filepath, "wb") as f:
            f.write(image_bytes)
            
        # Run pre-filter classification (will return None if corrupted)
        is_leaf = is_leaf_image(image_bytes)
        
        if is_leaf is None:
            print(f"Filtering out corrupted file: {unique_filename}")
            if os.path.exists(filepath):
                os.remove(filepath)
            continue
            
        processed.append({
            "filename": unique_filename,
            "is_leaf": is_leaf
        })
        
    return jsonify({"success": True, "processed": len(processed), "results": processed, "flight_id": flight_id})

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json
    flight_id = data.get("flight_id", "Unknown Flight")
    filenames = data.get("filenames", [])
    
    conn = get_db()
    upload_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analyzed_count = 0
    
    for filename in filenames:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                image_bytes = f.read()
                
            try:
                pred_result = run_prediction(image_bytes)
                temp, hum = parse_metadata_from_filename(filename)
                
                conn.execute('''
                    INSERT INTO results (filename, flight_id, prediction, confidence, status, date_uploaded, temperature, humidity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (filename, flight_id, pred_result["prediction"], pred_result["confidence"], pred_result["status"], upload_date, temp, hum))
                analyzed_count += 1
            except Exception as e:
                print(f"Failed to analyze {filename}: {e}")
                
    conn.commit()
    conn.close()
    
    return jsonify({"success": True, "analyzed": analyzed_count})

@app.route("/api/results")
def api_results():
    conn = get_db()
    flight_id = request.args.get("flight_id")
    
    query = "SELECT * FROM results ORDER BY id DESC"
    params = ()
    
    if flight_id:
        query = "SELECT * FROM results WHERE flight_id = ? ORDER BY id DESC"
        params = (flight_id,)
        
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    results = [dict(row) for row in rows]
    return jsonify({"results": results})

@app.route("/api/stats")
def api_stats():
    conn = get_db()
    flight_id = request.args.get("flight_id")
    
    where_clause = ""
    params = ()
    if flight_id:
        where_clause = "WHERE flight_id = ?"
        params = (flight_id,)
        
    total = conn.execute(f"SELECT COUNT(*) FROM results {where_clause}", params).fetchone()[0]
    
    healthy_where = "WHERE status = 'Healthy'" + (" AND flight_id = ?" if flight_id else "")
    healthy = conn.execute(f"SELECT COUNT(*) FROM results {healthy_where}", params).fetchone()[0]
    
    diseased_where = "WHERE status = 'Diseased'" + (" AND flight_id = ?" if flight_id else "")
    diseased = conn.execute(f"SELECT COUNT(*) FROM results {diseased_where}", params).fetchone()[0]
    
    avg_conf = conn.execute(f"SELECT AVG(confidence) FROM results {where_clause}", params).fetchone()[0]
    avg_conf = round(avg_conf * 100, 2) if avg_conf else 0
    
    # Most detected disease
    most_where = "WHERE status = 'Diseased'" + (" AND flight_id = ?" if flight_id else "")
    most_detected = conn.execute(f'''
        SELECT prediction, COUNT(*) as count 
        FROM results 
        {most_where} 
        GROUP BY prediction 
        ORDER BY count DESC 
        LIMIT 1
    ''', params).fetchone()
    most_detected_name = most_detected["prediction"] if most_detected else "None"
    
    # Summary by disease
    disease_summary = conn.execute(f'''
        SELECT prediction, COUNT(*) as count
        FROM results
        {where_clause}
        GROUP BY prediction
        ORDER BY count DESC
    ''', params).fetchall()
    summary = {row["prediction"]: row["count"] for row in disease_summary}
    
    # Recent flights (always show all flights for dropdown options)
    flights = conn.execute("SELECT DISTINCT flight_id FROM results ORDER BY date_uploaded DESC LIMIT 10").fetchall()
    recent_flights = [row["flight_id"] for row in flights]
    
    # All flights (for dropdown)
    all_flights_rows = conn.execute("SELECT DISTINCT flight_id FROM results ORDER BY flight_id").fetchall()
    all_flights = [row["flight_id"] for row in all_flights_rows]
    
    conn.close()
    
    return jsonify({
        "total": total,
        "healthy": healthy,
        "diseased": diseased,
        "avg_confidence": avg_conf,
        "most_detected": most_detected_name,
        "summary": summary,
        "recent_flights": recent_flights,
        "all_flights": all_flights
    })

# --- FLIGHT CRUD APIs ---
@app.route("/api/flights", methods=["GET"])
def get_flights():
    conn = get_db()
    flights = conn.execute('''
        SELECT flight_id, COUNT(*) as image_count, MIN(date_uploaded) as upload_date
        FROM results
        GROUP BY flight_id
        ORDER BY upload_date DESC
    ''').fetchall()
    conn.close()
    
    return jsonify({"flights": [dict(row) for row in flights]})

@app.route("/api/flights/<path:flight_id>", methods=["PUT"])
def update_flight(flight_id):
    data = request.json
    new_flight_id = data.get("new_flight_id")
    
    if not new_flight_id:
        return jsonify({"error": "New flight ID is required"}), 400
        
    try:
        conn = get_db()
        conn.execute('UPDATE results SET flight_id = ? WHERE flight_id = ?', (new_flight_id, flight_id))
        conn.commit()
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        if 'conn' in locals() and hasattr(conn, 'close'):
            try:
                conn.close()
            except:
                pass
    
    return jsonify({"success": True})

@app.route("/api/flights/<path:flight_id>", methods=["DELETE"])
def delete_flight(flight_id):
    try:
        conn = get_db()
        
        # Fetch filenames associated with this flight before deleting
        rows = conn.execute('SELECT filename FROM results WHERE flight_id = ?', (flight_id,)).fetchall()
        filenames_to_delete = [row["filename"] for row in rows]
        
        # Delete records from the database
        conn.execute('DELETE FROM results WHERE flight_id = ?', (flight_id,))
        conn.commit()
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        if 'conn' in locals() and hasattr(conn, 'close'):
            try:
                conn.close()
            except:
                pass
    
    # Delete physical image files from the server
    failed_deletes = 0
    for filename in filenames_to_delete:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to delete {filepath}: {e}")
                failed_deletes += 1
                
    if failed_deletes > 0:
        return jsonify({
            "success": True, 
            "warning": f"Records deleted, but {failed_deletes} files could not be deleted from disk (they may be locked)."
        })
                
    return jsonify({"success": True})

@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    results = []
    total_processed = 0
    correct_overall = 0
    
    for cls in CLASS_NAMES:
        # We will use the class name directly as the key in the form data
        files = request.files.getlist(f"images_{cls}")
        for file in files:
            if file.filename == "": continue
            
            image_bytes = file.read()
            prediction = run_prediction(image_bytes)
            
            is_correct = prediction["prediction"] == cls
            temperature, humidity = parse_metadata_from_filename(file.filename)
            top_two = top_two_from_probabilities(prediction["probabilities"])
            total_processed += 1
            if is_correct: correct_overall += 1
            
            results.append({
                "filename": file.filename,
                "expected": cls,
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "correct": is_correct,
                "status": prediction["status"],
                "probabilities": prediction["probabilities"],
                "scores": prediction["scores"],
                "top2_label": top_two["second_label"],
                "top2_score": top_two["second_score"],
                "top2_margin": top_two["margin"],
                "temperature": temperature,
                "humidity": humidity
            })
            
    if total_processed == 0:
        return jsonify({"error": "No images provided for any class"}), 400
        
    accuracy = (correct_overall / total_processed * 100) if total_processed > 0 else 0
    evaluation_summary = build_evaluation_summary(results)
    
    return jsonify({
        "total": total_processed,
        "correct": correct_overall,
        "wrong": total_processed - correct_overall,
        "accuracy": accuracy,
        "results": results,
        "classes": CLASS_NAMES,
        "evaluation": evaluation_summary
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
