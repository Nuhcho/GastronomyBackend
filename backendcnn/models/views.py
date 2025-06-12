import json
import os
import pickle
import joblib
import tensorflow as tf
from django.http import HttpResponseBadRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from tensorflow.keras.preprocessing import image
import xgboost
import sklearn
from sklearn.preprocessing import StandardScaler

def is_image_file(filename):
    image_extensions = {'.png', '.jpg'}
    ext = os.path.splitext(filename)[1].lower()
    return ext in image_extensions

def handle_uploaded_file(f, filename, id):
    downloads_dir = os.path.join('downloads')
    os.makedirs(downloads_dir, exist_ok=True)
    unique_filename = f"{id}_{filename}"
    file_path = os.path.join(downloads_dir, unique_filename)

    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

    print(f"Saved image to: {file_path}")
    return file_path  # Return the actual path


upload_counter = 0

@csrf_exempt
def reset(request):
    global upload_counter
    upload_counter = 0
    downloads_dir = os.path.join('downloads')
    if os.path.exists(downloads_dir):
        for filename in os.listdir(downloads_dir):
            file_path = os.path.join(downloads_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                return JsonResponse({"status": "error", "message": str(e)})
        os.rmdir(downloads_dir)
    else:
        return JsonResponse({"status": "error", "message": "Downloads directory does not exist."})
    return JsonResponse({"status": "success", "message": "Counter reset successfully."})

@csrf_exempt
def hello(request):
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        filename = uploaded_file.name
        if not is_image_file(filename):
            return HttpResponseBadRequest("Only png files are accepted.\n")

        global upload_counter
        upload_counter += 1

        # Save file with ID-prefixed name and get full path
        file_path = handle_uploaded_file(uploaded_file, filename, upload_counter)

        prediction = runBinary(file_path)  # Pass exact file path

        if prediction is None or "error" in prediction:
            return JsonResponse({
                "status": "error",
                "message": prediction.get("error", "Unknown error.")
            }, status=500)

        return JsonResponse({
            "status": "success",
            "id": upload_counter,
            "prediction": prediction
        })

    return HttpResponseBadRequest("Only POST with a file is accepted.\n")


def runBinary(file_path):
    try:
        # Load feature extractor and SVM models
        feature_extractor = tf.keras.models.load_model('backendcnn/models/resnet50_feature_extractor.h5')
        print("Feature extractor loaded successfully.")
        with open('backendcnn/models/svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        print("SVM model loaded successfully using pickle.")
        # Preprocess image
        img = image.load_img(file_path, target_size=(224, 224), color_mode='rgb')
        print(f"Image loaded and resized to 224x224: {file_path}")
        img_array = image.img_to_array(img)
        print("Image converted to array.")
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        print("Image preprocessed for ResNet50.")
        img_array = np.expand_dims(img_array, axis=0)
        print("Image expanded to 4D tensor.")

        # Feature extraction and classification
        features = feature_extractor.predict(img_array)
        print("Features extracted from image.")
        scaler = joblib.load('backendcnn/models/scaler.pkl')
        features = scaler.transform(features)
        print("Features scaled using StandardScaler.")
        print("Features shape:", features.shape)
        prediction = svm_model.predict(features)
        print("Prediction made by SVM model.")
        print("Prediction output:", prediction)
        print("Prediction shape:", prediction.shape)

        class_label = "Normal" if prediction[0] == 1 else "Abnormal"
        print(f"Class label determined: {class_label}")
        return runModel(file_path, class_label)

    except Exception as e:
        return {"error": str(e)}


def runModel(file_path, class_label):
    CLASS_NAMES = ['ADI', 'MUS', 'NOR', 'DEB', 'LYM', 'MUC', 'STR', 'TUM']
    try:
        # Load feature extraction model
        full_model = tf.keras.models.load_model('backendcnn/models/ResNet50_finetuned_hmu_V2.keras')
        resnet_feature_extractor = tf.keras.Model(
            inputs=full_model.input,
            outputs=full_model.get_layer('features_dense').output  # Adjust if needed
        )
        print("ResNet50 feature extractor loaded successfully.")

        # Load trained SVM model
        svm2_model = joblib.load('backendcnn/models/ResNet50_SVM_model.pkl')
        print("SVM2 model loaded successfully.")

        # Load and preprocess image
        img = image.load_img(file_path, target_size=(448, 448))
        img_array = image.img_to_array(img)
        patch_size = 74  # approx 448 / 6
        counts = np.zeros(len(CLASS_NAMES))

        # Divide image into 6x6 patches
        for row in range(6):
            for col in range(6):
                patch = img_array[row * patch_size:(row + 1) * patch_size,
                                  col * patch_size:(col + 1) * patch_size, :]
                patch = tf.image.resize(patch, (224, 224)).numpy()
                patch = tf.keras.applications.resnet50.preprocess_input(patch)
                patch = np.expand_dims(patch, axis=0)

                features = resnet_feature_extractor.predict(patch, verbose=0)
                features = np.ascontiguousarray(features, dtype=np.float32)

                prediction = svm2_model.predict(features)
                counts[prediction[0]] += 1

        # Calculate class distribution in percentages
        percentages = (counts / 36.0 * 100)

        # Build the output dictionary in the requested format
        result_output = {
            "Normal or Abnormal": class_label,
            "class_label": {}
        }
        for i, label in enumerate(CLASS_NAMES):
            result_output["class_label"][label] = f"{percentages[i]:.2f}%"

        return result_output

    except Exception as e:
        return {"error": str(e)}


@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        my_param = data.get('id')
        return JsonResponse(
            {
                "status": "success",
                "id": my_param,
                "classification": runModel(my_param)
            }
        )
    return JsonResponse({"status": "need post request"})