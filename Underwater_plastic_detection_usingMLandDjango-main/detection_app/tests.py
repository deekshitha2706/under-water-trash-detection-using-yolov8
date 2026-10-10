from django.test import TestCase
from pathlib import Path

import cv2

from .views import detect_and_annotate


class ImageDetectionTests(TestCase):
    def test_onnx_detector_annotates_an_image(self):
        sample_path = (
            Path(__file__).resolve().parent.parent
            / "static"
            / "images"
            / "aquarium-5650174_1280.jpg"
        )
        image = cv2.imread(str(sample_path))

        self.assertIsNotNone(image)
        result = detect_and_annotate(image)

        self.assertEqual(result.shape, image.shape)
        self.assertTrue((result != image).any())
