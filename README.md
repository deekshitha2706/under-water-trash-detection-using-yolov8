# Underwater Plastic Detection using Machine Learning and Django

A web app that detects **plastic waste in underwater images and videos**. Upload a file in the browser, a trained **YOLOv8** model finds the plastic, and the app shows the original next to the same file with detection boxes drawn on it.

## Links

| | |
|---|---|
| **GitHub repository** | https://github.com/deekshitha2706/Underwater_plastic_detection_usingMLandDjango |
| **Live website** | https://under-water-trash-detection.onrender.com |

> Marine plastic pollution harms ocean life and is hard to survey by hand. This project shows how an object-detection model can automate that first step.

---

## Features

- **Image detection:** upload an image and get an annotated result image.
- **Video detection:** upload a video; every frame is analysed and a new annotated video is produced.
- **Side-by-side view:** original and detected result are shown together.
- **Stored history:** uploads and results are saved through Django models (SQLite) and viewable in the Django admin.
- **Simple interface:** Bootstrap-styled pages, no setup beyond Python.

## How it works

```
Browser upload  ->  Django view  ->  YOLOv8 (model/best.pt)  ->  boxes drawn with OpenCV  ->  result page
```

1. The user uploads an image or video on the home page.
2. Django saves the file under `media/uploads/` and records it in the database.
3. `detection_app/views.py` loads the trained weights `model/best.pt` with Ultralytics YOLO.
4. For an image the model runs once; for a video it runs on every frame.
5. Detections are drawn onto the image or frames with OpenCV and saved under `media/results/`.
6. The result page shows the original and the annotated output.

## Tech stack

| Part | Technology |
|---|---|
| Web framework | Django |
| Object detection | Ultralytics YOLOv8 (PyTorch) |
| Image / video processing | OpenCV, NumPy |
| Database | SQLite |
| Front end | HTML templates + Bootstrap |

## Project structure

```
Underwater_plastic_detection_usingMLandDjango/
  manage.py
  db.sqlite3                      SQLite database
  Underwater_plastic_detection/   Django project (settings, urls, wsgi/asgi)
  detection_app/                  main app
    models.py                     DetectionImage, DetectionVideo
    forms.py                      upload forms
    views.py                      index, process_image, process_video
    urls.py                       routes
  templates/detection_app/        index.html, image_result.html, video_result.html
  model/best.pt                   trained YOLOv8 weights
  media/uploads/                  uploaded files
  media/results/                  annotated outputs
  static/
```

## Getting started

### Requirements
- Python 3.9 or newer
- pip
- A few GB of free disk space (PyTorch is large)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/deekshitha2706/Underwater_plastic_detection_usingMLandDjango.git
cd Underwater_plastic_detection_usingMLandDjango

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install django ultralytics opencv-python numpy pillow

# 4. Create database tables
python manage.py migrate

# 5. Start the server
python manage.py runserver
```

Open **http://127.0.0.1:8000/** in your browser.

### Usage
1. Choose an image under **Upload Image** and press **Process Image**, or a video under **Upload Video** and press **Process Video**.
2. Wait for processing. Videos take longer, because every frame is analysed.
3. View the original and the detected result, then press **Process Another Image/Video**.

Optional: `python manage.py createsuperuser` and visit `/admin/` to browse stored uploads.

## Model

- **Architecture:** YOLOv8 object detector, loaded from `model/best.pt`.
- **Training data, classes, and metrics:** *add your dataset name, number of images, class names, and results (precision, recall, mAP) here.*

## Limitations

- Videos are processed on the CPU unless a CUDA GPU is available, so long videos can be slow.
- Results are written with the `mp4v` codec, which some browsers cannot play inline. If the result video does not show, download it or switch the codec in `views.py` to `avc1`.
- Detection quality depends on water clarity, lighting and how close the training data is to your footage.
- The model is loaded on every request, which is simple but slower than loading it once.
- Uploaded files are never deleted automatically, so `media/` grows over time.

## Before deploying publicly

The settings are meant for local development. For any public deployment:
- Set `DEBUG = False` and fill in `ALLOWED_HOSTS`.
- Move `SECRET_KEY` to an environment variable and replace the one in the repo.
- Serve media and static files properly (for example WhiteNoise or a storage service) and use a production server such as gunicorn.
- Limit upload size and file types.

## Future improvements

- Show confidence scores and a count of detected plastic items.
- Real-time detection from a camera or live stream.
- More classes (nets, bottles, bags) and a larger, more varied dataset.
- Cache the model between requests and process videos in a background task.
- Export a detection report (CSV/PDF).

## License

This project is licensed under the MIT License. See the `LICENSE` file.
