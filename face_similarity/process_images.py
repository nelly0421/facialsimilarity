import os
import cv2
import numpy as np
import dlib
from facenet_pytorch import InceptionResnetV1
import torch
from django.conf import settings
from .models import Face
import base64

# 初始化 Dlib 的人脸检测器
detector = dlib.get_frontal_face_detector()

# 加载特征提取模型
model = InceptionResnetV1(pretrained='vggface2').eval()

def crop_and_extract_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        # 仅裁剪第一张检测到的人脸
        face = faces[0]
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cropped_img = img[y:y+h, x:x+w]
        cropped_img = cv2.resize(cropped_img, (160, 160), interpolation=cv2.INTER_LANCZOS4)
        img_tensor = torch.tensor(cropped_img).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            feature = model(img_tensor.unsqueeze(0)).numpy().flatten()
        return feature
    else:
        print("No face detected in", image_path)
        return None

def process_images():
    image_dir = '/face_similarity/images'
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        feature_vector = crop_and_extract_features(image_path)
        if feature_vector is not None:
            feature_vector_encoded = base64.b64encode(feature_vector).decode('utf-8')
            face = Face(image=image_name, feature_vector=feature_vector_encoded)
            face.save()
            print(f"Processed and saved features for {image_name}")

if __name__ == "__main__":
    process_images()
