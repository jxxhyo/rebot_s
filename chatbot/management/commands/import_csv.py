# import_csv.py
import csv
from django.core.management.base import BaseCommand
from chatbot.models import ResImage

class Command(BaseCommand):
    help = 'Import data from CSV file to ResImage model'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The path to the CSV file to be imported')

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 5:  # Ensure there are enough columns
                    image_name, image_en_url, image_ko_url, image_zh_url, image_ja_url = row[0], row[1], row[2], row[3], row[4]
                    ResImage.objects.create(
                        image_name=image_name, 
                        image_en_url=image_en_url, 
                        image_ko_url=image_ko_url, 
                        image_zh_url=image_zh_url, 
                        image_ja_url=image_ja_url
                    )
        
        self.stdout.write(self.style.SUCCESS(f'Data from {csv_file} successfully imported to ResImage model'))
