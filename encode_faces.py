import face_recognition
import os
import pickle

DATABASE_DIR = "database"
PKL_FILE = "face_encodings.pkl"

def encode_faces():
    encodings = []
    names = []

    for file in os.listdir(DATABASE_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(DATABASE_DIR, file)
            image = face_recognition.load_image_file(path)
            face_enc = face_recognition.face_encodings(image)

            if face_enc:
                encodings.append(face_enc[0])
                names.append(os.path.splitext(file)[0])

    with open(PKL_FILE, "wb") as f:
        pickle.dump((encodings, names), f)

    return encodings, names
