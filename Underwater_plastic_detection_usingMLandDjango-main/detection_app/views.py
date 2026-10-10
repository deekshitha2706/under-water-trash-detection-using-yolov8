from django.shortcuts import render,redirect
from django.conf import settings
from .forms import ImageUploadForm, VideoUploadForm
from functools import lru_cache
import logging
import os
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_yolo_model():
    import torch

    torch.set_num_threads(1)
    model_path = os.path.join(settings.BASE_DIR, 'model', 'best.pt')
    return YOLO(model_path)


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
                results = get_yolo_model().predict(
                    source=img_path,
                    device='cpu',
                    imgsz=640,
                    workers=0,
                    verbose=False,
                )
                os.makedirs(os.path.dirname(result_img_path), exist_ok=True)
                if not cv2.imwrite(result_img_path, results[0].plot()):
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
            
            model = get_yolo_model()
            
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
                results = model.predict(
                    source=frame,
                    device='cpu',
                    imgsz=640,
                    workers=0,
                    verbose=False,
                )
                
                # Plot results on frame
                res_plotted = results[0].plot()
                
                # Write the frame to the output video
                out.write(res_plotted)
            
            # Release resources
            cap.release()
            out.release()
            
            # Update the model with the result video path
            video_obj.result_video = f'results/result_{os.path.basename(video_path)}'
            video_obj.save()
            
            return render(request, 'detection_app/video_result.html', {'video_obj': video_obj})
    
    return redirect('index')

# Create your views here.
