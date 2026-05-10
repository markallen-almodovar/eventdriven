# 🐾 Cats and Dogs Classifier + Student Management System

> **Laboratory Activity: Final Project** — Web App with Machine Learning (FastAPI + React)
> **Course:** ITE03 (Web Systems & Technologies) + EVENTDP (Event-Driven Programming)
> **Section:** BSIT-III / BSCS-III
> **Team:** Mark Allen Almodovar · Jan Deive Marinas · Mykeah Jasmie Serrano · Reignce Dela Pena

---

## Project Overview

A unified full-stack web application combining two subject requirements into one platform:

| Subject | Component | Description |
|---------|-----------|-------------|
| **EVENTDP** | Cats and Dogs Classifier | User uploads a pet image → FastAPI processes it using a MobileNetV2 ML model → Cat/Dog prediction with confidence % is displayed |
| **ITE03** | Student Management System | Full-stack CRUD web app for managing student enrollment records, secured with JWT authentication |

---

## How It Works (ML Flow)

```
User uploads image
      ↓
React frontend sends image to FastAPI (POST /pets/predict)
      ↓
FastAPI preprocesses image (resize 160×160, normalize to [0,1])
      ↓
MobileNetV2 model runs inference
      ↓
Returns JSON: { "label": "Cat", "confidence": 91.08 }
      ↓
React displays prediction result
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, Vite 7, Tailwind CSS 3, React Router v7, Recharts |
| **ML Backend** | Python, FastAPI 0.115, TensorFlow 2.15, Keras 3.12, MobileNetV2, Pillow |
| **API Backend** | Node.js, Express.js, MySQL2, bcryptjs, jsonwebtoken |
| **Database** | MySQL (via XAMPP) |

---

## Prerequisites

Before running the project, make sure you have:

- **Node.js** 18 or higher — https://nodejs.org
- **Python** 3.10 or higher — https://python.org
- **XAMPP** (MySQL only, Apache not needed) — https://apachefriends.org
- **Git** — https://git-scm.com

---

## Installation & Setup (First Time Only)

### Step 1 — Clone the repository

```bash
git clone https://github.com/MarkAllen-Almodovar/eventdriven.git
cd eventdriven
```

### Step 2 — Install frontend dependencies

```bash
npm install
```

### Step 3 — Install backend dependencies

```bash
cd database
npm install
cd ..
```

### Step 4 — Set up Python virtual environment

```bash
python -m venv venv_new
venv_new\Scripts\activate
pip install -r requirements.txt
```

### Step 5 — Configure environment variables

Copy the example env file and fill in your values:

```bash
copy .env.example .env
```

Root `.env` (Vite frontend):
```
VITE_EXPRESS_API=http://localhost:5000
VITE_FASTAPI_URL=http://127.0.0.1:8000
```

Create `database/.env`:
```
PORT=5000
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=studentdb
JWT_SECRET=your_long_random_secret_here
```

### Step 6 — Set up the database

1. Open **XAMPP Control Panel** and start **MySQL**
2. Run the database setup script:

```bash
mysql -u root -e "source database/init.sql"
```

Or open **phpMyAdmin** (`http://localhost/phpmyadmin`) and import `database/init.sql`.

### Step 7 — Train the ML model (only needed once)

```bash
venv_new\Scripts\python.exe train_pets_model.py
```

> This takes approximately 15–20 minutes on CPU. The model will be saved to `saved_model/pets_classification_model.keras`.

### Step 8 — Convert model for compatibility (only needed once)

```bash
venv_new\Scripts\python.exe convert_models.py
venv_new\Scripts\python.exe patch_pets_h5.py
```

> This converts the model to `.h5` format for stable loading on Windows.

---

## Running the Project

Start **three terminals** simultaneously:

### Terminal 1 — Frontend (React + Vite)

```bash
npm run dev
```

Runs on: **http://localhost:5173**

### Terminal 2 — Express Backend (Student Management API)

```bash
cd database
node server.js
```

Runs on: **http://localhost:5000**

### Terminal 3 — ML Server (FastAPI + TensorFlow)

```bash
venv_new\Scripts\python.exe -W ignore -m uvicorn mlserver:app --port 8000
```

Runs on: **http://127.0.0.1:8000**

> ⏳ The ML server takes **60–90 seconds** to load the model on first startup. The server is available immediately at port 8000 but will return a loading message until the model is ready.

### Or use the batch file (Windows)

```bash
START_SERVERS.cmd
```

Double-click `START_SERVERS.cmd` to open all three servers in separate windows automatically.

---

## Open the App

Once all three servers are running, open your browser:

**http://localhost:5173**

---

## API Endpoints

### FastAPI — ML Server (port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check — returns model status |
| GET | `/health` | Detailed model info + class labels |
| **POST** | **`/predict`** | **Upload pet image → returns Cat/Dog + confidence %** |
| POST | `/pets/predict` | Same as `/predict` (primary endpoint) |
| POST | `/pets/upload` | Alias for `/pets/predict` |

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/pets/predict \
  -F "file=@cat.jpg"
```

**Example response:**
```json
{
  "label": "Cat",
  "confidence": 91.08
}
```

### Express API — Student Management (port 5000)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/register` | — | Register new user |
| POST | `/auth/login` | — | Login, returns JWT token |
| GET | `/students` | JWT | Get all students |
| GET | `/students/:id` | JWT | Get one student |
| POST | `/students` | JWT | Add student |
| PUT | `/students/:id` | JWT | Update student |
| DELETE | `/students/:id` | JWT | Delete student |

---

## ML Model Details

- **Architecture:** MobileNetV2 (pretrained on ImageNet) + custom classification head
- **Input size:** 160×160 RGB pixels, normalized to [0, 1]
- **Output:** Cat or Dog with confidence percentage
- **Training:** Two-phase transfer learning
  - Phase 1: Train classification head only (base frozen)
  - Phase 2: Fine-tune top 53 MobileNetV2 layers
- **Validation accuracy:** **91.08%**
- **Test accuracy:** **90.17%**
- **Dataset:** Kaggle Dogs vs Cats — ~2,611 images total

---

## Project Structure

```
├── src/                         # React frontend
│   ├── App.jsx                  # Routes
│   ├── pages/
│   │   ├── LandingPage.jsx      # Home page
│   │   ├── ClassifyPetsPage.jsx # Cats vs Dogs classifier (upload + webcam)
│   │   ├── auth/                # Login + Register pages
│   │   └── students/            # Student CRUD + Dashboard pages
│   ├── layout/NavBar.jsx        # Sidebar navigation
│   └── context/AuthContext.jsx  # JWT auth state
│
├── database/                    # Express backend
│   ├── server.js                # REST API routes
│   ├── mysql.js                 # MySQL connection
│   └── init.sql                 # Database setup script
│
├── mlserver.py                  # FastAPI server (POST /predict)
├── classify_pets.py             # ML model inference
├── train_pets_model.py          # Model training script
├── convert_models.py            # Model format conversion
├── patch_pets_h5.py             # Model compatibility patch
│
├── saved_model/
│   ├── pets_classification_model_fixed.h5  # Runtime model
│   └── pets_class_names.json              # Class labels
│
├── requirements.txt             # Python dependencies
├── package.json                 # Node.js dependencies
├── .env.example                 # Environment variable template
└── START_SERVERS.cmd            # Windows: start all servers
```

---

## Features

### 🐾 Cats and Dogs Classifier
- Upload image via drag-and-drop, file picker, or webcam capture
- Instant Cat / Dog prediction with confidence percentage
- Low-confidence warning (< 60%)
- Classification history (last 5 results with thumbnails)

### 🎓 Student Management System
- JWT authentication (register, login, auto-logout on token expiry)
- Full CRUD: add, view, edit, delete student records
- Real-time search, sortable columns, pagination
- Export to CSV
- Analytics dashboard with enrollment charts

---

## Team

| Name | Role |
|------|------|
| Mark Allen Almodovar | Lead Developer |
| Jan Deive Marinas | Developer |
| Mykeah Jasmie Serrano | Developer |
| Reignce Dela Pena | Developer |

**Course/Section:** BSIT-III / BSCS-III | ITE03 + EVENTDP Finals Project
