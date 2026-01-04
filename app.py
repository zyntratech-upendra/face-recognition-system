from flask import Flask, render_template, request, redirect, session, jsonify, Response
import os, json, pickle, cv2, base64
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image
import face_recognition

from auth import USERS
from encode_faces import encode_faces

app = Flask(__name__)
app.secret_key = "secretkey"

DATABASE_DIR = "database"
PKL_FILE = "face_encodings.pkl"
ATTENDANCE_FILE = "attendance.json"
THRESHOLD = 0.45

os.makedirs(DATABASE_DIR, exist_ok=True)

# ---------- GLOBAL FRAME ----------
latest_frame = None

# ---------- LOAD FACE ENCODINGS ----------
if os.path.exists(PKL_FILE):
    with open(PKL_FILE, "rb") as f:
        known_encodings, known_names = pickle.load(f)
else:
    known_encodings, known_names = encode_faces()

# ---------- SAVE ATTENDANCE ----------
def save_attendance(name, mode):
    with open(ATTENDANCE_FILE, "r") as f:
        records = json.load(f)

    records.append({
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "mode": mode
    })

    with open(ATTENDANCE_FILE, "w") as f:
        json.dump(records, f, indent=4)

# ---------- LOGIN ----------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        user = USERS.get(u)
        if user and user["password"] == p:
            session["username"] = u
            session["role"] = user["role"]
            return redirect("/admin" if user["role"] == "admin" else "/user")

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

# ---------- ADMIN DASHBOARD ----------
@app.route("/admin")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect("/")

    with open(ATTENDANCE_FILE, "r") as f:
        records = json.load(f)

    summary = {}
    for r in records:
        summary[r["name"]] = summary.get(r["name"], 0) + 1

    return render_template("admin_dashboard.html", summary=summary)

# ---------- USER DASHBOARD ----------
@app.route("/user")
def user_dashboard():
    if session.get("role") != "user":
        return redirect("/")

    username = session["username"]

    with open(ATTENDANCE_FILE, "r") as f:
        records = json.load(f)

    user_records = [r for r in records if r["name"] == username]

    return render_template(
        "user_dashboard.html",
        name=username,
        records=user_records
    )

# ---------- ADMIN IMAGE UPLOAD ----------
@app.route("/admin/upload", methods=["POST"])
def admin_upload():
    if session.get("role") != "admin":
        return redirect("/")

    files = request.files.getlist("images")
    for file in files:
        file.save(os.path.join(DATABASE_DIR, file.filename))

    global known_encodings, known_names
    known_encodings, known_names = encode_faces()

    return redirect("/admin")

# ---------- LAPTOP CAMERA PREVIEW ----------
def gen_frames():
    cap = cv2.VideoCapture(0)
    global latest_frame

    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        latest_frame = frame.copy()

        # 🔹 Resize for faster processing
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)

        for (top, right, bottom, left), enc in zip(locations, encodings):
            dists = face_recognition.face_distance(known_encodings, enc)
            name = "Unknown"
            color = (0, 0, 255)

            if len(dists) and min(dists) < THRESHOLD:
                name = known_names[np.argmin(dists)]
                color = (0, 255, 0)

            # Scale back to original size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

    cap.release()


@app.route("/take-attendance")
def take_attendance():
    return render_template("camera.html")

@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------- SUBMIT ATTENDANCE ----------
@app.route("/submit-attendance")
def submit_attendance():
    global latest_frame

    if latest_frame is None:
        return redirect("/take-attendance")

    rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    marked = set()

    for enc in encodings:
        dists = face_recognition.face_distance(known_encodings, enc)
        if len(dists) == 0:
            continue

        best = np.argmin(dists)
        if dists[best] < THRESHOLD:
            name = known_names[best]

            # USER: face must match username
            if session.get("role") == "user":
                if name == session.get("username"):
                    marked.add(name)

            # ADMIN: allow all
            elif session.get("role") == "admin":
                marked.add(name)

    for name in marked:
        save_attendance(name, "laptop")

    return redirect("/admin" if session.get("role") == "admin" else "/user")

# ---------- PHONE ATTENDANCE ----------
@app.route("/phone-attendance")
def phone_attendance():
    return render_template("phone.html")

@app.route("/submit-photo", methods=["POST"])
def submit_photo():
    data = request.json["image"]
    img_data = base64.b64decode(data.split(",")[1])
    img = Image.open(BytesIO(img_data)).convert("RGB")
    np_img = np.array(img)

    small = cv2.resize(np_img, (0, 0), fx=0.5, fy=0.5)
    encs = face_recognition.face_encodings(small)

    if not encs:
        return jsonify({"status": "no_face"})

    enc = encs[0]
    dists = face_recognition.face_distance(known_encodings, enc)

    if len(dists) == 0 or min(dists) >= THRESHOLD:
        return jsonify({"status": "unknown"})

    name = known_names[np.argmin(dists)]

    if session.get("role") == "user":
        if name != session.get("username"):
            return jsonify({"status": "mismatch"})
        save_attendance(name, "phone")

    elif session.get("role") == "admin":
        save_attendance(name, "phone")

    return jsonify({"status": "matched"})

@app.route("/after-attendance")
def after_attendance():
    return redirect("/admin" if session.get("role") == "admin" else "/user")

@app.route("/attendance/<name>")
def person_attendance(name):
    if session.get("role") != "admin":
        return redirect("/")

    with open(ATTENDANCE_FILE, "r") as f:
        records = json.load(f)

    person_records = [r for r in records if r["name"] == name]

    return render_template(
        "person_attendance.html",
        name=name,
        records=person_records
    )

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

