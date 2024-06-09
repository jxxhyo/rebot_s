import csv
from django.core.management.base import BaseCommand
from chatbot.models import Restaurant

class Command(BaseCommand):
    help = 'Load and update restaurant data from CSV'

    def handle(self, *args, **kwargs):
        file_path = '/Users/jaehyo/Documents/projects/chatbot/django_chatbot/all_res_info_df.csv'  # Update to the correct CSV file path

        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    restaurant = Restaurant.objects.get(name=row['식당 이름'])
                    restaurant.mood = row['분위기 추천']
                    restaurant.save()
                    self.stdout.write(self.style.SUCCESS(f'Successfully updated {restaurant.name}'))
                except Restaurant.DoesNotExist:
                    self.stdout.write(self.style.ERROR(f'Restaurant {row["식당 이름"]} does not exist'))

        self.stdout.write(self.style.SUCCESS('Successfully loaded data'))
