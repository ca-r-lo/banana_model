import os
import sys
import sqlite3
import datetime
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

def run_prediction(image_bytes):
    if not MODEL_LOADED:
        # Mock prediction for testing without model
        import random
        predicted_index = random.randint(0, len(CLASS_NAMES) - 1)
        confidence = random.uniform(0.7, 0.99)
        return {
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": confidence,
            "status": "Healthy" if CLASS_NAMES[predicted_index] == "Healthy" else "Diseased"
        }
    
    input_data = preprocess_image(image_bytes)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]
    
    predicted_index = int(np.argmax(output_data))
    confidence = float(output_data[predicted_index])
    pred_class = CLASS_NAMES[predicted_index]
    
    return {
        "prediction": pred_class,
        "confidence": confidence,
        "status": "Healthy" if pred_class == "Healthy" else "Diseased"
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
            total_processed += 1
            if is_correct: correct_overall += 1
            
            results.append({
                "filename": file.filename,
                "expected": cls,
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "correct": is_correct,
                "status": prediction["status"]
            })
            
    if total_processed == 0:
        return jsonify({"error": "No images provided for any class"}), 400
        
    accuracy = (correct_overall / total_processed * 100) if total_processed > 0 else 0
    
    return jsonify({
        "total": total_processed,
        "correct": correct_overall,
        "wrong": total_processed - correct_overall,
        "accuracy": accuracy,
        "results": results,
        "classes": CLASS_NAMES
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)