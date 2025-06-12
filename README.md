# CancerCNN Backend

This is the backend server for the CancerCNN project, built with Django and TensorFlow. It provides REST API endpoints for image upload, classification, and model-based predictions using deep learning and SVM models.

---

## Features

- **Image Upload & Classification:** Upload `.png` or `.jpg` images for binary and multi-class classification.
- **Patch-based Analysis:** Images are divided into patches for detailed class distribution.
- **Model Integration:** Uses TensorFlow (ResNet50) and SVM models for feature extraction and classification.
- **CORS Support:** Configured for cross-origin requests from a frontend (e.g., React).

---

## Project Structure

```
backendcnn/
│
├── backendcnn/
│   ├── models/
│   │   ├── views.py         # Main API logic
│   │   └── __init__.py
│   ├── downloads/           # Uploaded images (runtime)
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── ...
├── db.sqlite3
├── manage.py
├── README.md
└── .gitignore
```

---

### Prerequisites

- Python 3.8+
- pip
- TensorFlow
- scikit-learn
- xgboost
- Django 5.2+
- django-cors-headers

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/Nuhcho/GastronomyBackend
    cd backendcnn
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On Linux/Mac
    ```

3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is missing, install the packages listed in Prerequisites manually.)*

4. **Place model files:**
    - Place your `.keras`, `.h5`, and `.pkl` model files in `backendcnn/models/` as referenced in the code.

5. **Run migrations:**
    ```sh
    python manage.py migrate
    ```

6. **Start the server:**
    ```sh
    python manage.py runserver
    ```

---

## API Endpoints

### 1. `/model/` (POST)

**Description:** Upload an image for binary and multi-class classification.

**Request:**
- Method: `POST`
- Form-data: `file` (image file, `.png` or `.jpg`)

**Response:**
```json
{
  "status": "success",
  "id": 1,
  "prediction": {
    "Normal or Abnormal": "Normal",
    "class_label": {
      "ADI": "0.00%",
      "MUS": "0.00%",
      "...": "..."
    }
  }
}
```

---

### 2. `/predict/` (POST)

**Description:** Run multi-class classification on an already uploaded image by its file path.

**Request:**
- Method: `POST`
- Body (JSON): `{ "id": "<file_path>" }`

**Response:**
```json
{
  "status": "success",
  "id": "<file_path>",
  "classification": {
    "Normal or Abnormal": "Normal",
    "class_label": {
      "ADI": "0.00%",
      "MUS": "0.00%",
      "...": "..."
    }
  }
}
```

---

### 3. `/reset/` (POST)

**Description:** Reset the upload counter and delete all files in the `downloads/` directory.

**Request:**  
- Method: `POST`

**Response:**
```json
{
  "status": "success",
  "message": "Counter reset successfully."
}
```

---

## Notes

- The backend expects model files (`.keras`, `.h5`, `.pkl`) to be present in `backendcnn/models/`.
- These models can be found in the Google Drive linked with the report
- The `downloads/` directory is used for storing uploaded images temporarily.
- CORS is enabled for `http://localhost:3000` by default (see `settings.py`).
