import tensorflow as tf
import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ======================================
# 1. Wczytywanie danych
# ======================================

def load_voc_data(image_dir, annotation_dir, classes):
    images = []
    boxes = []
    labels = []

    for annotation_file in os.listdir(annotation_dir):
        if not annotation_file.endswith('.xml'):
            continue

        # Parsowanie pliku XML
        tree = ET.parse(os.path.join(annotation_dir, annotation_file))
        root = tree.getroot()

        # Wczytanie obrazu
        filename = root.find("filename").text
        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

        image_boxes = []
        image_labels = []

        # Pobieranie bounding boxów i etykiet
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in classes:
                continue

            class_id = classes.index(class_name)

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            image_boxes.append([xmin, ymin, xmax, ymax])
            image_labels.append(class_id)

        boxes.append(image_boxes)
        labels.append(image_labels)

    return np.array(images), np.array(boxes, dtype=object), np.array(labels, dtype=object)


# ======================================
# 2. Przygotowanie danych
# ======================================

def preprocess_data(images, boxes, labels, input_size, num_classes):
    processed_images = []
    processed_boxes = []
    processed_labels = []

    for i in range(len(images)):
        img = images[i]
        h, w, _ = img.shape

        # Resize obrazu
        img_resized = cv2.resize(img, (input_size, input_size))
        processed_images.append(img_resized / 255.0)

        # Normalizacja współrzędnych bounding boxów
        normalized_boxes = []
        for box in boxes[i]:
            xmin, ymin, xmax, ymax = box
            normalized_boxes.append([
                xmin / w, ymin / h, xmax / w, ymax / h
            ])
        processed_boxes.append(normalized_boxes)

        # One-hot encoding etykiet
        one_hot_labels = tf.keras.utils.to_categorical(labels[i], num_classes=num_classes)
        processed_labels.append(one_hot_labels)

    return np.array(processed_images), np.array(processed_boxes, dtype=object), np.array(processed_labels, dtype=object)


# ======================================
# 3. Tworzenie modelu
# ======================================

def create_model(input_shape, num_classes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
    base_model.trainable = False  # Zamrożenie wag

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_classes + 4)  # num_classes dla klas, 4 dla bboxów
    ])
    return model


# ======================================
# 4. Funkcja strat
# ======================================

def custom_loss(y_true, y_pred):
    y_true_boxes, y_true_classes = y_true[..., :4], y_true[..., 4:]
    y_pred_boxes, y_pred_classes = y_pred[..., :4], y_pred[..., 4:]

    box_loss = tf.reduce_mean(tf.square(y_true_boxes - y_pred_boxes))  # MSE dla bbox
    class_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_classes, y_pred_classes))  # CE dla klas

    return box_loss + class_loss


# ======================================
# 5. Podział danych na treningowe i walidacyjne
# ======================================

def split_data(images, boxes, labels, val_split=0.2):
    # Użycie train_test_split z scikit-learn
    train_images, val_images, train_boxes, val_boxes, train_labels, val_labels = train_test_split(
        images, boxes, labels, test_size=val_split, random_state=42
    )

    return (train_images, train_boxes, train_labels), (val_images, val_boxes, val_labels)


# ======================================
# 6. Trenowanie modelu
# ======================================

def train_model(model, train_data, val_data, epochs=10):
    model.compile(optimizer='adam', loss=custom_loss)

    # Rozdzielenie danych
    train_images, train_boxes, train_labels = train_data
    val_images, val_boxes, val_labels = val_data

    # Łączenie bounding boxów i klas w jedną tablicę
    train_targets = [np.hstack((train_boxes[i], train_labels[i])) for i in range(len(train_boxes))]
    val_targets = [np.hstack((val_boxes[i], val_labels[i])) for i in range(len(val_boxes))]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_targets)).batch(32).shuffle(1000)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_targets)).batch(32)

    # Trening
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    return history


# ======================================
# 7. Wizualizacja predykcji
# ======================================

def predict_and_visualize(model, image, classes):
    input_image = cv2.resize(image, (224, 224)) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    predictions = model.predict(input_image)
    predicted_boxes = predictions[..., :4][0]
    predicted_classes = predictions[..., 4:][0]

    h, w, _ = image.shape
    for i, box in enumerate(predicted_boxes):
        class_id = np.argmax(predicted_classes[i])
        score = predicted_classes[i][class_id]
        if score > 0.5:
            x_min, y_min, x_max, y_max = [int(coord * dim) for coord, dim in zip(box, [w, h, w, h])]
            label = classes[class_id]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.imshow(image)
    plt.axis("off")
    plt.show()


# ======================================
# Główna część programu
# ======================================

if __name__ == "__main__":
    # Ustawienia
    IMAGE_DIR = "images/"  # Folder z obrazami
    ANNOTATION_DIR = "etykietowane/"  # Folder z etykietami XML
    CLASSES = ["car", "truck", "bus"]  # Twoje klasy
    INPUT_SIZE = 224
    NUM_CLASSES = len(CLASSES)
    EPOCHS = 10

    # Wczytywanie danych
    images, boxes, labels = load_voc_data(IMAGE_DIR, ANNOTATION_DIR, CLASSES)

    # Podział danych na treningowe i walidacyjne
    (train_images, train_boxes, train_labels), (val_images, val_boxes, val_labels) = split_data(images, boxes, labels)

    # Przetwarzanie danych
    train_data = preprocess_data(train_images, train_boxes, train_labels, INPUT_SIZE, NUM_CLASSES)
    val_data = preprocess_data(val_images, val_boxes, val_labels, INPUT_SIZE, NUM_CLASSES)

    # Tworzenie modelu
    model = create_model(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), num_classes=NUM_CLASSES)

    # Trenowanie modelu
    history = train_model(model, train_data, val_data, epochs=EPOCHS)

    # Testowanie predykcji
    sample_image = images[0]  # Możesz wybrać dowolny obraz z testowego zbioru
    predict_and_visualize(model, sample_image, CLASSES)