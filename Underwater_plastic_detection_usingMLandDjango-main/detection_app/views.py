from django.shortcuts import render,redirect
from django.conf import settings
from .forms import ImageUploadForm, VideoUploadForm
from functools import lru_cache
import logging
import os
import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

CLASS_NAMES = (
    'Mask', 'can', 'cellphone', 'electronics', 'gbottle', 'glove', 'metal',
    'misc', 'net', 'pbag', 'pbottle', 'plastic', 'rod', 'sunglasses', 'tire',
)
MODEL_SIZE = 640

@lru_cache(maxsize=1)
def get_inference_session():
    model_path = os.path.join(settings.BASE_DIR, 'model', 'best.onnx')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=['CPUExecutionProvider'],
    )


def detect_and_annotate(frame):
    height, width = frame.shape[:2]
    ratio = min(MODEL_SIZE / width, MODEL_SIZE / height)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    horizontal_padding = MODEL_SIZE - resized_width
    vertical_padding = MODEL_SIZE - resized_height
    left = round(horizontal_padding / 2 - 0.1)
    top = round(vertical_padding / 2 - 0.1)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        vertical_padding - top,
        left,
        horizontal_padding - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    session = get_inference_session()
    input_name = session.get_inputs()[0].name
    detections = session.run(
        None,
        {input_name: tensor},
    )[0][0]

    annotated = frame.copy()
    for x1, y1, x2, y2, confidence, class_id in detections:
        if confidence < 0.25:
            continue
        x1 = max(0, min(width - 1, round((x1 - left) / ratio)))
        y1 = max(0, min(height - 1, round((y1 - top) / ratio)))
        x2 = max(0, min(width - 1, round((x2 - left) / ratio)))
        y2 = max(0, min(height - 1, round((y2 - top) / ratio)))
        color = (
            (37 * int(class_id) + 80) % 256,
            (17 * int(class_id) + 160) % 256,
            (29 * int(class_id) + 220) % 256,
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f'{CLASS_NAMES[int(class_id)]} {confidence:.2f}'
        cv2.putText(
            annotated,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return annotated


def index(request):
    image_form = ImageUploadForm()
    video_form = VideoUploadForm()
    return render(request, 'detection_app/index.html', {'image_form': image_form, 'video_form': video_form})

def process_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_obj = form.save()
            img_path = img_obj.image.path
            result_img_path = os.path.join(
                settings.MEDIA_ROOT,
                'results',
                f'result_{os.path.basename(img_path)}',
            )
            try:
                frame = cv2.imread(img_path)
                if frame is None:
                    raise ValueError('The uploaded file could not be decoded as an image.')
                os.makedirs(os.path.dirname(result_img_path), exist_ok=True)
                if not cv2.imwrite(result_img_path, detect_and_annotate(frame)):
                    raise OSError(f'Could not write detection result to {result_img_path}')
                img_obj.result_image = f'results/{os.path.basename(result_img_path)}'
                img_obj.save()
            except Exception:
                logger.exception('Image detection failed for upload %s', img_obj.pk)
                raise

            return render(request, 'detection_app/image_result.html', {'img_obj': img_obj})
    
    return redirect('index')

def process_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_obj = form.save()
            
            # Process the video
            video_path = video_obj.video.path
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create result video writer
            result_video_path = os.path.join(settings.MEDIA_ROOT, 'results', f'result_{os.path.basename(video_path)}')
            os.makedirs(os.path.dirname(result_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' if mp4v doesn't work
            out = cv2.VideoWriter(result_video_path, fourcc, fps, (width, height))
            
            # Process frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLOv8 inference on the frame
                # Write the frame to the output video
                out.write(detect_and_annotate(frame))
            
            # Release resources
            cap.release()
            out.release()
            
            # Update the model with the result video path
            video_obj.result_video = f'results/result_{os.path.basename(video_path)}'
            video_obj.save()
            
            return render(request, 'detection_app/video_result.html', {'video_obj': video_obj})
    
    return redirect('index')

# Create your views here.
