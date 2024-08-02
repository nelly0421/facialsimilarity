import os
import cv2
import dlib
import torch
import base64
from facenet_pytorch import InceptionResnetV1

# 設置 Django 環境
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "face_similarity.settings")
import django
django.setup()

from ranking.models import Face

# 加載預訓練的 ArcFace 模型
model = InceptionResnetV1(pretrained='vggface2').eval()

# 加載 Dlib 的人臉檢測器
detector = dlib.get_frontal_face_detector()

def extract_features(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))  # Resize to 160x160
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize
    with torch.no_grad():
        feature = model(img.unsqueeze(0)).numpy().flatten()
    return feature

def detect_and_crop_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None

    # 假設只取第一張檢測到的人臉
    x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
    cropped_face = img[y:y+h, x:x+w]
    return cropped_face

# 圖片目錄
image_dir = '/ssd2/nelly/face_similarity/images'
processed_dir = '/ssd2/nelly/face_similarity/processed_images'

# 創建保存裁剪後人臉圖像的目錄
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# 遍歷並處理所有圖片
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    cropped_face = detect_and_crop_face(image_path)
    if cropped_face is None:
        print(f"No face detected in {image_name}. Skipping.")
        continue

    # 保存裁剪後的人臉圖像
    processed_image_path = os.path.join(processed_dir, image_name)
    cv2.imwrite(processed_image_path, cropped_face)

    feature_vector = extract_features(cropped_face)
    feature_vector_bytes = base64.b64encode(feature_vector)

    # 保存到數據庫
    face = Face(name=image_name, image=processed_image_path, feature_vector=feature_vector_bytes)
    face.save()
    print(f'Successfully uploaded and processed {image_name}')
