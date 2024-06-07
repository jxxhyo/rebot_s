import os
import django
from django.core.files import File

# Django 프로젝트 설정 파일 경로
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_chatbot.settings')  # 'my_django_project'를 실제 프로젝트 이름으로 변경
django.setup()

from chatbot.models import ResImage, Restaurant  # 'my_app'을 실제 앱 이름으로 변경
ResImage.objects.all().delete()

# 파일들이 저장된 루트 디렉토리 설정
root_dir = 'dalle-image'

# 디렉토리 트리 순회
for restaurant_name in os.listdir(root_dir):
    restaurant_path = os.path.join(root_dir, restaurant_name)
    if os.path.isdir(restaurant_path):
        # 레스토랑 객체 가져오기 (없는 경우 생성)
        restaurant, created = Restaurant.objects.get_or_create(name=restaurant_name)
        
        # ResImage 객체 가져오기 (없는 경우 생성)
        res_image, created = ResImage.objects.get_or_create(restaurant=restaurant, image_name=restaurant_name)

        # 이미지 파일들 순회
        for image_type in ['chi_image', 'eng_image', 'jap_image', 'kor_image']:
            image_path = os.path.join(restaurant_path, image_type)
            if os.path.exists(image_path) and os.path.isdir(image_path):
                for image_file in os.listdir(image_path):
                    if image_file.endswith('.jpg'):  # 확장자는 필요에 따라 조정
                        file_path = os.path.join(image_path, image_file)

                        # 이미지 필드에 파일 추가
                        with open(file_path, 'rb') as f:
                            if image_type == 'chi_image':
                                res_image.image_zh.save(image_file, File(f), save=False)
                            elif image_type == 'eng_image':
                                res_image.image_en.save(image_file, File(f), save=False)
                            elif image_type == 'jap_image':
                                res_image.image_ja.save(image_file, File(f), save=False)
                            elif image_type == 'kor_image':
                                res_image.image_ko.save(image_file, File(f), save=False)

        # 객체 저장
        res_image.save()
