from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .models import Face
from facenet_pytorch import InceptionResnetV1
import torch
import cv2
import numpy as np
import base64
from scipy.spatial.distance import cosine
from django.conf import settings
import os
from PIL import Image
import dlib
from io import BytesIO

# 初始化 Dlib 的人脸检测器
detector = dlib.get_frontal_face_detector()

# 加载预训练的模型
model = InceptionResnetV1(pretrained='vggface2').eval()

def find_similar_faces(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(image.name, image)
        image_path = fs.path(filename)

        # 对上传的图像进行人脸检测和裁剪
        feature_vector = crop_and_extract_features(image_path)

        if feature_vector is not None:
            all_faces = Face.objects.all()
            similarities = []

            for face in all_faces:
                stored_vector = np.frombuffer(base64.b64decode(face.feature_vector), dtype=np.float32)
                similarity = 1 - cosine(feature_vector, stored_vector)
                similarities.append((face, similarity))

            # 按相似度排序并删除重复的人物
            sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            top_5_faces = []
            added_names = set()  # 用于存储已添加的人物名称

            for face_obj, similarity in sorted_similarities:
                if face_obj.name not in added_names:
                    added_names.add(face_obj.name)  # 标记此人物已添加
                    image_path = os.path.join(settings.IMAGE_ROOT, face_obj.image.name)
                    print(image_path)
                    image = Image.open(image_path)
                    image = image.convert('RGB') if image.mode != 'RGB' else image
                    image = image.resize((200, 200), Image.BICUBIC)

                    buffer = BytesIO()
                    image.save(buffer, format="JPEG")
                    image_bytes = buffer.getvalue()
                    image_base64 = base64.b64encode(image_bytes).decode()

                    image_url = f'data:image/jpeg;base64,{image_base64}'
                    top_5_faces.append({'rank': len(top_5_faces) + 1, 'face': face_obj, 'similarity': similarity, 'image_url': image_url})

                    if len(top_5_faces) >= 5:  # 只选前五个
                        break

            return render(request, 'ranking/results.html', {
                'top_5': top_5_faces,
                'uploaded_file_url': fs.url(filename)
            })

    return render(request, 'ranking/home.html')


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
        print("No face detected in uploaded image")
        return None
