from django.shortcuts import render,redirect
from django.conf import settings
from .forms import ImageUploadForm, VideoUploadForm
from .models import DetectionImage, DetectionVideo
import os
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

def index(request):
    image_form = ImageUploadForm()
    video_form = VideoUploadForm()
    return render(request, 'detection_app/index.html', {'image_form': image_form, 'video_form': video_form})

def process_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_obj = form.save()
            
            # Load the YOLOv8 model
            model_path = os.path.join(settings.BASE_DIR, 'model', 'best.pt')
            model = YOLO(model_path)
            
            # Process the image
            img_path = img_obj.image.path
            results = model(img_path)
            
            # Save the result image
            result_img_path = os.path.join(settings.MEDIA_ROOT, 'results', f'result_{os.path.basename(img_path)}')
            res_plotted = results[0].plot()
            cv2.imwrite(result_img_path, res_plotted)
            
            # Update the model with the result image path
            img_obj.result_image = f'results/result_{os.path.basename(img_path)}'
            img_obj.save()
            
            return render(request, 'detection_app/image_result.html', {'img_obj': img_obj})
    
    return redirect('index')

def process_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_obj = form.save()
            
            # Load the YOLOv8 model
            model_path = os.path.join(settings.BASE_DIR, 'model', 'best.pt')
            model = YOLO(model_path)
            
            # Process the video
            video_path = video_obj.video.path
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create result video writer
            result_video_path = os.path.join(settings.MEDIA_ROOT, 'results', f'result_{os.path.basename(video_path)}')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' if mp4v doesn't work
            out = cv2.VideoWriter(result_video_path, fourcc, fps, (width, height))
            
            # Process frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLOv8 inference on the frame
                results = model(frame)
                
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
