import cv2
import os
import numpy as np

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer.yml")

# ================= LOAD FACE DETECTOR =================
face_net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)


# =====================================================
# =============== FACE DETECTION ======================
# =====================================================
def detect_faces(frame):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    boxes = []

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]

        if conf > 0.6:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            boxes.append(box.astype(int))

    return boxes


# =====================================================
# =============== DATA COLLECTION =====================
# =====================================================
def collect_faces(name):

    save_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print("[INFO] Press S to save face")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_faces(frame)

        for (x1, y1, x2, y2) in boxes:

            face = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))

            cv2.imshow("Face", gray)

            key = cv2.waitKey(1)

            if key == ord('s'):
                count += 1
                cv2.imwrite(
                    f"{save_dir}/{count}.jpg",
                    gray
                )
                print(f"Saved {count}")

        if count >= 30 or cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =====================================================
# ================== TRAIN MODEL ======================
# =====================================================
def train_model():

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}

    current_id = 0

    for person in os.listdir(DATASET_DIR):

        label_map[person] = current_id
        person_path = os.path.join(DATASET_DIR, person)

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)

            face = cv2.imread(img_path,
                              cv2.IMREAD_GRAYSCALE)

            faces.append(face)
            labels.append(current_id)

        current_id += 1

    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_PATH)

    print("[INFO] Training Completed")


# =====================================================
# ========= OBSTRUCTION DETECTION =====================
# =====================================================
def lower_face_covered(face_gray):

    h = face_gray.shape[0]

    upper = face_gray[:int(h * 0.45), :]
    lower = face_gray[int(h * 0.55):, :]

    upper_edges = cv2.Canny(upper, 50, 150)
    lower_edges = cv2.Canny(lower, 50, 150)

    if np.count_nonzero(upper_edges) == 0:
        return False

    return (
        np.count_nonzero(lower_edges)
        < 0.4 * np.count_nonzero(upper_edges)
    )


# =====================================================
# ================= RECOGNITION =======================
# =====================================================
def run_security():

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_faces(frame)

        for (x1, y1, x2, y2) in boxes:

            face = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(face,
                                cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))

            label, dist = recognizer.predict(gray)

            if dist < 90:
                if lower_face_covered(gray):
                    text = "YOU (COVERED)"
                    color = (0, 255, 255)
                else:
                    text = "YOU"
                    color = (0, 255, 0)
            else:
                text = "UNKNOWN"
                color = (0, 0, 255)

            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          color, 2)

            cv2.putText(frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color, 2)

        cv2.imshow("Face Security", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =====================================================
# ===================== MENU ==========================
# =====================================================
print("""
1 → Collect Face Dataset
2 → Train Model
3 → Run Security System
""")

choice = input("Select option: ")

if choice == "1":
    name = input("Enter name: ")
    collect_faces(name)

elif choice == "2":
    train_model()

elif choice == "3":
    run_security()