# Project Context — Cats vs Dogs Classifier + Student Management System

> Paste this file into a new Kiro chat to restore full context.
> Last updated: May 7, 2026

---

## What this is

A unified React web app combining two school subjects into one final project.
Both subjects are under the same instructor who merged the requirements.

- **EVENTDP** → Cats vs Dogs AI Classifier (MobileNetV2 transfer learning)
- **ITE03** → Student Management System (full-stack CRUD)

**The gender classifier was removed.** Only Cats vs Dogs + Student Management remain.

---

## Two features in one app

### 1. Cats vs Dogs Classifier (EVENTDP)
- Upload a pet image OR use webcam → FastAPI → MobileNetV2 model → Cat/Dog + confidence %
- Drag & drop or click to upload, image preview, confidence bar
- Webcam mode: live video, capture frame, classify
- Low-confidence warning (< 60%) with amber styling
- Classification history log (last 5 results with thumbnails)
- FastAPI runs on port 8000, endpoint: `/pets/predict`
- Model: `saved_model/pets_classification_model_fixed.h5` (patched h5 for Keras 3 compat)
- Labels: `saved_model/pets_class_names.json` → `{"cats_set": 0, "dogs_set": 1}`
- Accuracy: 91.08% val, 90.17% test
- Training: two-phase MobileNetV2 (Phase 1: head only, Phase 2: unfreeze top 53 layers)
- Preprocessing: resize 160×160, divide by 255 — NO autocontrast

### 2. Student Management System (ITE03)
- Full CRUD for student records with JWT authentication (8h token, auto-logout)
- Fields: Student No., Full Name, Email, Course, Year Level
- Register/Login → protected routes → view/add/edit/delete students
- Real-time search across all fields
- Sortable columns (click header to toggle asc/desc)
- Pagination (10 per page)
- Export to CSV
- Analytics Dashboard: stat cards + donut chart (by course) + bar chart (by year level)
- Express runs on port 5000, MySQL database: `studentdb`

---

## Tech Stack

- **Frontend:** React 19 + Vite 7 + Tailwind CSS 3 + React Router v7 + Recharts
- **ML Backend:** FastAPI 0.115 + TensorFlow 2.15 + Keras 3.12 + MobileNetV2 + Pillow
- **SMS Backend:** Express.js + MySQL2 + bcryptjs + JWT (`database/server.js`)
- **Database:** MySQL via XAMPP (tables: `users`, `students`)

---

## Design

- Dark theme: `slate-950` background, `slate-900` cards
- Orange accent (`orange-500/600`) for AI/pets classifier
- Violet accent (`violet-600`) for student management + auth
- Collapsible sidebar nav
- Toast notifications (no alert() calls)
- Animated skeleton loading rows for the students table
- Animated 404 page with 5-second countdown auto-redirect
- Responsive layout

---

## Active Routes

| Path | Component | Auth |
|------|-----------|------|
| `/` | LandingPage | — |
| `/classify-pets` | ClassifyPetsPage | — |
| `/login` | LoginPage | — |
| `/register` | RegisterPage | — |
| `/students` | StudentsPage | JWT |
| `/students/add` | AddStudentPage | JWT |
| `/students/edit/:id` | EditStudentPage | JWT |
| `/students/dashboard` | DashboardPage | JWT |
| `/students/:id` | StudentProfilePage | JWT |

Note: `/classify` (gender) route is REMOVED. `ClassifyPage.jsx` still exists in src/pages/ but is not imported or routed.

---

## Key Files

```
src/App.jsx                          # routes + ProtectedRoute
src/main.jsx                         # entry point + AuthProvider
src/config/api.js                    # API URLs + COURSES list
src/context/AuthContext.jsx          # JWT auth state + auto-expiry timer
src/components/Toast.jsx             # toast notifications + useToast hook
src/components/SkeletonRow.jsx       # animated skeleton for table loading
src/layout/NavBar.jsx                # dark sidebar layout (no gender link)
src/pages/LandingPage.jsx            # hero + about + how it works + features + visual + developer
src/pages/ClassifyPetsPage.jsx       # pets classifier (upload + webcam tabs + history)
src/pages/ClassifyPage.jsx           # UNUSED — gender classifier (kept but not routed)
src/pages/auth/LoginPage.jsx
src/pages/auth/RegisterPage.jsx
src/pages/students/StudentsPage.jsx
src/pages/students/AddStudentPage.jsx
src/pages/students/EditStudentPage.jsx
src/pages/students/DashboardPage.jsx
src/pages/students/StudentProfilePage.jsx

database/server.js                   # Express REST API
database/mysql.js                    # MySQL pool (reads .env)
database/init.sql                    # DB setup + seed data

mlserver.py                          # FastAPI server (active entry point)
ml-server.py                         # OLD — kept for reference, not used
classify_pets.py                     # pets model inference (uses fixed h5)
classify.py                          # gender model inference (not called by active routes)

train_pets_model.py                  # MobileNetV2 two-phase training
evaluate_pets_model.py               # test-set evaluation
convert_models.py                    # .keras → .h5 conversion
patch_pets_h5.py                     # removes 'groups' key for Keras 3 compat

saved_model/pets_classification_model_fixed.h5   # ACTIVE runtime model
saved_model/pets_class_names.json                # {"cats_set": 0, "dogs_set": 1}
saved_model/gender_classification_model.keras    # legacy, not used by active routes
```

---

## How to Start

```bash
# Terminal 1 — Frontend (port 5173)
npm run dev

# Terminal 2 — Express backend (port 5000, needs MySQL running in XAMPP)
cd database && node server.js

# Terminal 3 — ML server (port 8000, models load in ~60-90s background thread)
venv_new\Scripts\python.exe -W ignore -m uvicorn mlserver:app --port 8000
```

Or double-click `START_SERVERS.cmd`.

---

## Known Issues / Notes

- **ML server startup:** TF 2.15 on Windows hangs if models are loaded at import time (JSON dump bug). Fixed by loading models in a FastAPI `startup` event using `asyncio.run_in_executor`. Server responds at port 8000 immediately; models ready in ~60-90s.
- **Model format:** Pets model saved with Keras 2.15 (has `groups:1` in DepthwiseConv2D). Patched to `_fixed.h5` for Keras 3 compatibility. Gender model saved with Keras 3.12.1 — loaded via `keras.saving.load_model`.
- **FastAPI version:** Must be 0.115.12. Older versions (0.104.x) are incompatible with pydantic v2.
- **GPU:** RTX 3050 present but CUDA not installed. Training runs on CPU. Install CUDA 11.8 + cuDNN 8.6 for GPU acceleration.
- **MySQL:** Must be started in XAMPP before running Express. Express continues without DB but student features won't work.
- **Vite port:** Configured to 5173 with `strictPort: false` fallback. If 5173 is taken it uses 5174.

---

## Express API

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | /auth/register | No | Register new user |
| POST | /auth/login | No | Login, returns JWT |
| GET | /students | JWT | Get all students |
| GET | /students/:id | JWT | Get one student |
| POST | /students | JWT | Add student |
| PUT | /students/:id | JWT | Update student |
| DELETE | /students/:id | JWT | Delete student |

## FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check + models_ready flag |
| GET | /health | Model info + input size + class labels |
| POST | /pets/predict | Upload pet image → label + confidence |
| POST | /pets/upload | Alias for /pets/predict |

---

## Team

- Mark Allen Almodovar — Lead Developer
- Jan Deive Marinas — Developer
- Mykeah Jasmie Serrano — Developer
- Reignce Dela Pena — Developer

Course/Section: BSIT-III
