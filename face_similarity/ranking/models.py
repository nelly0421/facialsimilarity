from django.db import models

class Face(models.Model):
    name = models.CharField(max_length=255)
    image = models.ImageField(upload_to='images/')  # 確保圖片存儲在 media/images 文件夾中
    feature_vector = models.BinaryField()

    def __str__(self):
        return self.name
