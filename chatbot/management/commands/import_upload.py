import csv
from django.core.management.base import BaseCommand
from chatbot.models import BookmarkRestaurantInfo

class Command(BaseCommand):
    help = 'Import CSV data to BookmarkRestaurantInfo model'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The CSV file to import')

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']

        # 기존 데이터 삭제
        BookmarkRestaurantInfo.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Successfully deleted all existing data'))

        # 새로운 데이터 추가
        with open(csv_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                BookmarkRestaurantInfo.objects.create(
                    name_ko=row['name_ko'],
                    name_en=row['name_en'],
                    name_ja=row['name_ja'],
                    name_zh=row['name_zn'],
                    image_ko=row['image_ko'],
                    image_en=row['image_en'],
                    image_ja=row['image_ja'],
                    image_zh=row['image_zn'],
                    bookmark_count=row['bookmark_count'],
                    latitude=row['latitude'],
                    longitude=row['longitude']
                )

        self.stdout.write(self.style.SUCCESS('Successfully imported data'))
