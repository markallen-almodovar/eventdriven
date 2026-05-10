# Documentation Report
## Cats and Dogs Classifier + Student Management System

**Course:** ITE03 (Web Systems & Technologies) + EVENTDP (Event-Driven Programming)
**Section:** BSIT-III / BSCS-III
**Team:** Mark Allen Almodovar · Jan Deive Marinas · Mykeah Jasmie Serrano · Reignce Dela Pena

---

## Part 1: Project Overview

### Description

This project is a unified full-stack web application that combines the final requirements of two subjects — ITE03 and EVENTDP — into a single platform. It consists of two independent but integrated components: an AI-powered image classification system that identifies cats and dogs from uploaded photos, and a student management system for managing enrollment records with full CRUD functionality.

### Purpose of the Website

The website serves two distinct purposes:

**AI Component (EVENTDP):** To provide an accessible, browser-based tool that uses deep learning to classify pet images. A user uploads or captures a photo of a cat or dog, and the system instantly identifies the animal and returns a confidence percentage. No technical knowledge is required from the end user — the complexity of the neural network is hidden behind a simple drag-and-drop interface.

**Student Management Component (ITE03):** To provide a secure, centralized digital system for managing student enrollment records. Authorized users can register, log in, and perform full CRUD operations on student data, search and filter records in real time, export data to CSV, and view enrollment analytics through an interactive dashboard.

### Problem It Solves

The Cats and Dogs Classifier demonstrates how a trained machine learning model can be deployed as a practical web service — making deep learning accessible through a browser without requiring any local Python or ML setup from the user. It bridges the gap between a trained model and a real-world usable application.

The Student Management System solves the problem of manual, paper-based or spreadsheet-based student record keeping by providing a centralized, searchable, and secure digital system with role-based access control through JWT authentication.

---

## Part 2: Technologies Used

### Frontend Technologies

| Technology | Version | Role in the System |
|-----------|---------|-------------------|
| **React** | 19 | Component-based UI library — builds the entire frontend as a single-page application. Each page (classifier, student list, dashboard, etc.) is a React component with its own state. |
| **Vite** | 7 | Build tool and development server — compiles and bundles the React app, provides hot module replacement (HMR) for instant updates during development. |
| **Tailwind CSS** | 3 | Utility-first CSS framework — handles all styling including the dark theme, responsive layout, animations, and component styling without writing custom CSS files. |
| **React Router** | v7 | Client-side routing — manages navigation between pages (home, classifier, students, dashboard, login) without full page reloads, and protects authenticated routes. |
| **Recharts** | 2 | Chart library — renders the donut chart (students by course) and bar chart (students by year level) in the analytics dashboard. |

**Why these technologies?** React was chosen for its component reusability and reactive state management, which is essential for features like the live classification history sidebar, real-time search in the student table, and the confidence bar animation. Tailwind CSS allows rapid UI development without context-switching to separate CSS files. Vite was chosen over Create React App for its significantly faster startup and build times. React Router v7 provides clean URL-based navigation with built-in support for protected routes.

### Backend Technologies

| Technology | Version | Role in the System |
|-----------|---------|-------------------|
| **FastAPI** | 0.115 | Python web framework — serves as the ML API server. Receives uploaded images via HTTP POST, passes them through the classification model, and returns predictions as JSON. |
| **Uvicorn** | 0.34 | ASGI server — runs the FastAPI application, handles concurrent HTTP requests asynchronously. |
| **Express.js** | (Node.js) | JavaScript web framework — serves as the student management REST API. Handles user registration, login, and all student CRUD operations. |
| **MySQL** | (via XAMPP) | Relational database — stores user accounts (with hashed passwords) and student records with unique constraints on student number and email. |
| **bcryptjs** | — | Password hashing library — securely hashes user passwords before storing them in the database using the bcrypt algorithm with a salt factor of 10. |
| **jsonwebtoken** | — | JWT library — issues signed tokens on login and verifies them on every protected API request. Tokens expire after 8 hours. |

**Why these technologies?** FastAPI was chosen for the ML server because it is Python-native (same language as TensorFlow), asynchronous by default, and automatically generates interactive API documentation. Express.js was chosen for the student management backend because of its simplicity, large ecosystem, and the team's familiarity with JavaScript. MySQL was chosen because it is relational and well-suited for structured student records with enforced uniqueness constraints.

### Machine Learning Framework

| Technology | Version | Role in the System |
|-----------|---------|-------------------|
| **TensorFlow** | 2.15 | Core deep learning framework — provides the training infrastructure, data pipeline (tf.data), and model execution engine. |
| **Keras** | 3.12 | High-level neural network API — used to define the model architecture (MobileNetV2 + custom head), load saved models, and run inference. |
| **MobileNetV2** | (ImageNet weights) | Pre-trained CNN backbone — provides feature extraction learned from 1.2 million ImageNet images as the base of the classifier. |
| **Pillow** | 11.2 | Image processing library — opens uploaded image files, converts them to RGB, and resizes them to 160×160 pixels before inference. |
| **scikit-learn** | 1.6 | Machine learning utilities — generates the confusion matrix and classification report for model evaluation after training. |
| **NumPy** | 1.26 | Numerical computing — handles array operations for image preprocessing and prediction output processing. |

**Why these technologies?** TensorFlow and Keras are the industry standard for deep learning in Python with extensive documentation, community support, and production-ready deployment tools. MobileNetV2 was specifically chosen because it is a lightweight, efficient architecture designed for resource-constrained environments — it achieves high accuracy through transfer learning while being fast enough to run on CPU without requiring a GPU.

---

## Part 3: Website Structure

### Page 1: Home Page (`/`)

The landing page is the main entry point of the application. It is divided into seven sections:

**Hero Section**
Displays the project title "🐾 Cats and Dogs Classifier" in a large heading with an orange glow background effect. Contains two call-to-action buttons: "Try the Classifier →" (links to `/classify-pets`) and "Student Management" (links to `/register`). A badge at the top identifies the project as "ITE03 + EVENTDP · Finals Project."

**About Section**
Two side-by-side cards explaining each component:
- EVENTDP card (orange accent) — describes the Cats and Dogs Classifier, MobileNetV2 transfer learning, and the 91.08% validation accuracy achieved
- ITE03 card (indigo accent) — describes the Student Management System, JWT authentication, and the full-stack CRUD features

Subject badges at the bottom identify which subject each component belongs to.

**How It Works Section**
A five-step flow diagram showing the complete pipeline from image upload to prediction result, with a summary flow bar at the bottom showing: Upload Pet Image → FastAPI Receives File → Preprocessing → MobileNetV2 Inference → JSON Result → React Displays Prediction.

**Features Section**
Six feature cards covering: Cats and Dogs Classifier, Real-Time Classification, Webcam Capture Mode, JWT Authentication, Full Student CRUD, and Analytics Dashboard.

**Visual Section**
Mockup previews of the two main interfaces (classifier and student table) with fake browser chrome, plus a model workflow visualization diagram showing the data flow from Pet Image through Preprocessing → MobileNetV2 → Classification Head → Prediction.

**Developer Section**
Team member cards with names and roles, a project description paragraph, and course/section badges.

**Footer**
Tech stack pills listing all technologies used, copyright notice, and team names.

---

### Page 2: About Section

The About section is embedded within the Home Page (Section 2). It explains the project's dual purpose — the AI classification system for EVENTDP and the student management system for ITE03. It includes accuracy metrics, technology choices, and subject-to-feature mapping badges. There is no separate `/about` route; the landing page serves as the combined home and about page.

---

### Page 3: Features / Services Page

The Features section is also embedded within the Home Page (Section 4). It presents six feature cards:

1. **Cats and Dogs Classifier** — MobileNetV2 transfer learning, 1,000+ training images, instant prediction
2. **Real-Time Classification** — server-side processing through FastAPI, JSON response
3. **Webcam Capture Mode** — live camera feed, capture frame, classify directly from browser
4. **JWT Authentication** — secure login required for student management access
5. **Full Student CRUD** — create, read, update, delete with all required fields
6. **Analytics Dashboard** — stat cards, donut chart by course, bar chart by year level

---

### Page 4: Prediction Page (`/classify-pets`)

This is the main AI feature page — the Cats and Dogs Classifier.

**Header**
Page title "🐾 Cats and Dogs Classifier" with a subtitle and four model info badges: Model (MobileNetV2), Accuracy (91.08%), Input (160×160 px), Classes (Cat / Dog).

**Tab Switcher**
Two tabs: "Upload Image" and "Use Webcam."

**Upload Tab**
- Drag-and-drop zone with a 🐾 icon and instructions, or click to open file picker
- Accepts JPG, PNG, WEBP up to 5MB
- Image preview displayed after selection
- "Classify" button (orange) sends the image to FastAPI `/pets/predict`
- "Reset" button clears the current image and result
- Error message displayed if the server returns an error

**Webcam Tab**
- "Start Camera" button requests browser camera permission
- Live video feed with a red "LIVE" indicator badge
- "Capture" button freezes the frame and stops the stream
- "Classify This" button sends the captured frame to the API
- "Retake" button restarts the camera for a new capture
- "Stop" button turns off the camera without capturing

**Result Panel**
Appears below the upload/webcam area after classification:
- Large emoji + label: 🐱 Cat or 🐶 Dog
- Confidence progress bar (orange for Cat, blue for Dog, amber if confidence < 60%)
- Low-confidence warning message if below 60%: "Low confidence — try a clearer photo with the full animal visible"

**History Sidebar**
Appears on the right side after the first classification:
- Shows the last 5 results with thumbnail, label, and confidence percentage
- "Clear" button removes all history
- Amber confidence text for low-confidence results

---

### Additional Pages

**Login Page (`/login`)**
Username and password form with validation. On success, stores the JWT token and username in localStorage and redirects to `/students`. Displays inline error messages for invalid credentials.

**Register Page (`/register`)**
Username, password, and confirm password form. Minimum 6-character password requirement. On success, shows a green checkmark and redirects to login after 1.5 seconds.

**All Students (`/students`)** *(requires login)*
Paginated table with columns: Student No., Name, Email, Course, Year Level, Actions. Features real-time search across all fields, sortable columns (click header to toggle asc/desc), pagination (10 per page with smart ellipsis), CSV export button, and View/Edit/Delete actions per row. Delete shows a confirmation modal.

**Add Student (`/students/add`)** *(requires login)*
Form with fields: Student No., Full Name, Email Address, Course (dropdown with 10 options), Year Level (1–4). Validates all fields and shows inline error for duplicate student number or email.

**Edit Student (`/students/edit/:id`)** *(requires login)*
Same form as Add Student, pre-filled with the existing student's data.

**Student Profile (`/students/:id`)** *(requires login)*
Read-only view of a single student's details with Edit and Delete buttons.

**Dashboard (`/students/dashboard`)** *(requires login)*
Four stat cards: Total Students, Courses (count), Most Enrolled (course name), Avg per Course. Two charts: a donut/pie chart showing enrollment distribution by course, and a bar chart showing enrollment by year level (1st–4th Year). Animated skeleton loading state while data fetches.

**404 Page (`*`)**
Animated "404" display with a countdown timer that auto-redirects to the home page after 5 seconds.

---

## Part 4: Machine Learning Model

### Model Type

The model is a **Convolutional Neural Network (CNN)** built using **Transfer Learning** with **MobileNetV2** as the pre-trained backbone. Transfer learning is a technique where a model already trained on a large dataset (ImageNet — 1.2 million images, 1,000 classes) is adapted for a new, more specific task. Instead of training from scratch, the model reuses the feature extraction knowledge already embedded in MobileNetV2's weights.

### What the Model Predicts

Given any input image, the model predicts whether the image contains a **Cat** or a **Dog**, along with a **confidence percentage** (0–100%) indicating how certain the model is. The output is a softmax probability distribution over two classes: `[P(Cat), P(Dog)]`. The class with the higher probability is returned as the prediction.

### Model Architecture

```
Input Image
  └─ 160 × 160 × 3 (RGB, normalized to [0, 1])
         ↓
MobileNetV2 Base (pretrained on ImageNet)
  └─ 153 layers of depthwise separable convolutions
  └─ Extracts spatial features from the image
  └─ Output: feature map of shape (5, 5, 1280)
         ↓
GlobalAveragePooling2D
  └─ Reduces (5, 5, 1280) → (1280,) vector
         ↓
BatchNormalization
         ↓
Dense(256, activation='relu')
         ↓
Dropout(0.4)   ← prevents overfitting
         ↓
Dense(64, activation='relu')
         ↓
Dropout(0.2)
         ↓
Dense(2, activation='softmax')
  └─ Output: [P(Cat), P(Dog)]
```

**Total parameters:** ~2.6 million
**Trainable parameters (Phase 1):** ~347,000 (head only)
**Trainable parameters (Phase 2):** ~1.2 million (head + top MobileNetV2 layers)

### Training Strategy — Two-Phase Approach

The model was trained in two phases to maximize accuracy while preventing the destruction of pretrained ImageNet weights.

**Phase 1 — Classification Head Training**

The MobileNetV2 base is completely frozen (all 153 layers have `trainable=False`). Only the custom classification head (GlobalAveragePooling → BatchNorm → Dense → Dropout → Dense → Dropout → Dense) is trained. This allows the head to learn to interpret MobileNetV2's features without disrupting the pretrained weights.

- Optimizer: Adam, learning rate = 0.001
- Loss: Categorical Crossentropy
- Max epochs: 15
- Early stopping: patience = 5 (monitors val_loss, restores best weights)
- Checkpoint: saves best model by val_accuracy

**Phase 2 — Fine-Tuning**

The top 53 layers of MobileNetV2 (from layer index 100 onward) are unfrozen and trained together with the classification head. A much lower learning rate is used to make small, careful adjustments to the pretrained weights without destroying them.

- Optimizer: Adam, learning rate = 0.0001 (10× lower than Phase 1)
- Loss: Categorical Crossentropy
- Max epochs: 30
- Early stopping: patience = 7 (monitors val_loss, restores best weights)
- ReduceLROnPlateau: halves learning rate if val_loss doesn't improve for 3 epochs (minimum LR: 1e-7)
- Checkpoint: saves best model by val_accuracy

### Performance Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **91.08%** |
| Validation Loss | 0.2335 |
| Test Accuracy | **90.17%** |
| Test Loss | 0.2424 |

The model achieves over 90% accuracy on both the validation set (seen during training for monitoring) and the test set (completely unseen data), demonstrating good generalization.

---

## Part 5: Dataset / Images Used

### Source of Dataset

The dataset used is the **Kaggle Dogs vs. Cats dataset**, a publicly available benchmark dataset originally created for a Kaggle competition. It contains real-world photographs of cats and dogs collected from the internet, representing a wide variety of breeds, poses, backgrounds, lighting conditions, and image qualities.

**Dataset URL:** https://www.kaggle.com/c/dogs-vs-cats

### Number of Images / Data Samples

The dataset was split into three subsets for training, validation, and testing:

| Split | Cats | Dogs | Total |
|-------|------|------|-------|
| Training (`dogs_cats_split/train/`) | 500 | 500 | **1,000** |
| Validation (`dogs_cats_split/validation/`) | ~403 | ~404 | **~807** |
| Test (`dogs_cats_split/test/`) | ~402 | ~402 | **~804** |
| **Grand Total** | | | **~2,611** |

The images are organized into subdirectories named `cats_set/` and `dogs_set/` under each split folder. The folder names serve as the class labels — `tf.keras.utils.image_dataset_from_directory` automatically assigns class indices based on alphabetical folder order (`cats_set` → 0, `dogs_set` → 1).

### Type of Data

All data consists of **JPEG photographs** of real cats and dogs. The images vary significantly in:
- **Resolution** — different original sizes, all resized to 160×160 during preprocessing
- **Aspect ratio** — portrait, landscape, and square images
- **Background** — indoor, outdoor, plain, complex backgrounds
- **Lighting** — natural light, artificial light, shadows, overexposed
- **Subject position** — full body, close-up face, partial view, multiple animals
- **Breed** — multiple cat and dog breeds represented

This real-world variation makes the classification task more challenging and the trained model more generalizable to new, unseen images.

### Data Preprocessing

Before being fed into the model, each image goes through the following preprocessing steps (implemented in `classify_pets.py` for inference and `train_pets_model.py` for training):

**Step 1 — Open and Convert to RGB**
```python
img = Image.open(img_path).convert("RGB")
```
Ensures all images are in 3-channel color format regardless of the original format. This handles grayscale images (1 channel) and images with an alpha transparency channel (RGBA, 4 channels).

**Step 2 — Resize to 160×160 pixels**
```python
img = img.resize(IMG_SIZE, Image.LANCZOS)
```
All images are resized to 160×160 pixels to match MobileNetV2's expected input dimensions. LANCZOS resampling is used for high-quality downscaling that preserves edge sharpness.

**Step 3 — Normalize pixel values to [0, 1]**
```python
arr = np.array(img, dtype=np.float32) / 255.0
```
Pixel values are converted from the integer range [0, 255] to floating-point values in [0, 1]. This normalization is consistent between training (using `tf.keras.layers.Rescaling(1/255)`) and inference (dividing by 255.0 in Pillow), ensuring the model receives the same distribution of values it was trained on.

**Step 4 — Add batch dimension**
```python
return np.expand_dims(arr, axis=0)
```
Wraps the image array in a batch of size 1 (shape: `[1, 160, 160, 3]`) as required by the model's `predict()` method.

### Data Augmentation

Data augmentation was applied **during training only** (not during validation or test evaluation, and not during inference). Augmentation artificially increases the diversity of the training set by applying random transformations to each image in every epoch, helping the model generalize better and reducing overfitting.

The following augmentations were applied using Keras preprocessing layers inside the `tf.data` pipeline:

```python
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.12),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.15),
])
```

| Augmentation | Parameter | What It Does |
|-------------|-----------|-------------|
| **Random Horizontal Flip** | — | Randomly mirrors the image left-to-right (50% chance). Simulates a cat or dog facing either direction. |
| **Random Rotation** | factor=0.12 (≈ ±43°) | Randomly rotates the image by up to ±43 degrees. Simulates photos taken at different angles. |
| **Random Zoom** | factor=0.15 (±15%) | Randomly zooms in or out by up to 15%. Simulates different distances from the subject. |
| **Random Translation** | height=0.1, width=0.1 | Randomly shifts the image up/down and left/right by up to 10%. Simulates the subject not being centered. |
| **Random Brightness** | factor=0.15 (±15%) | Randomly adjusts image brightness. Simulates different lighting conditions. |
| **Random Contrast** | factor=0.15 (±15%) | Randomly adjusts image contrast. Simulates different camera settings and lighting. |

These augmentations are applied on-the-fly during training using `tf.data.AUTOTUNE` for parallel processing, meaning each epoch sees slightly different versions of the training images. The validation and test sets are never augmented — they use only normalization to provide an unbiased measure of model performance.

---

## Summary

| Component | Technology | Key Metric |
|-----------|-----------|-----------|
| Frontend | React 19 + Vite + Tailwind CSS | — |
| ML API | FastAPI + TensorFlow + Keras | Port 8000 |
| Student API | Express.js + MySQL | Port 5000 |
| ML Model | MobileNetV2 (Transfer Learning) | 91.08% val accuracy |
| Dataset | Kaggle Dogs vs Cats | ~2,611 images |
| Authentication | JWT (8-hour tokens) | bcrypt password hashing |

---

*Documentation prepared by: Mark Allen Almodovar, Jan Deive Marinas, Mykeah Jasmie Serrano, Reignce Dela Pena*
*Course: ITE03 + EVENTDP — Finals Project, BSIT-III / BSCS-III, 2026*
