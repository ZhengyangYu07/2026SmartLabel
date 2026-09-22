from django.test import TestCase
from django.contrib.auth.models import User

from .models import ImageResult, Task, TextResult
from .views import _get_label_verification_stats


class TextClassificationStatsTests(TestCase):
    def test_unverified_labeled_count_excludes_unlabeled_candidates(self):
        user = User.objects.create_user(username='stats-user', password='pass')
        task = Task.objects.create(
            user=user,
            name='text stats',
            task_type='text-classification',
            data_file='uploads/data/text.csv',
            uploaded_total=8,
            label_list=['a', 'b'],
        )

        TextResult.objects.bulk_create([
            TextResult(task=task, sequence_number=1, text_id='m1', content='manual 1', label='a', confidence=-1, status='unverified'),
            TextResult(task=task, sequence_number=2, text_id='m2', content='manual 2', label='b', confidence=-1, status='unverified'),
            TextResult(task=task, sequence_number=3, text_id='m3', content='manual 3', label='a', confidence=-1, status='verified'),
            TextResult(task=task, sequence_number=4, text_id='u1', content='candidate 1', label='a', confidence=0.61, status='unverified'),
            TextResult(task=task, sequence_number=5, text_id='u2', content='candidate 2', label='a', confidence=0.62, status='unverified'),
            TextResult(task=task, sequence_number=6, text_id='u3', content='candidate 3', label='b', confidence=0.63, status='unverified'),
            TextResult(task=task, sequence_number=7, text_id='u4', content='candidate 4', label='b', confidence=0.64, status='unverified'),
            TextResult(task=task, sequence_number=8, text_id='u5', content='candidate 5', label='a', confidence=0.65, status='unverified'),
        ])

        stats = _get_label_verification_stats(task, TextResult, 'task')

        self.assertEqual(stats['manual_annotated_count'], 3)
        self.assertEqual(stats['unverified_count'], 2)
        self.assertEqual(stats['verified_count'], 1)
        self.assertEqual(stats['unlabeled_count'], 5)
        self.assertEqual(stats['total_verifiable_count'], 3)
        self.assertEqual(stats['verification_progress'], 33.3)


class ImageClassificationStatsTests(TestCase):
    def test_unverified_labeled_count_excludes_unlabeled_candidates(self):
        user = User.objects.create_user(username='image-stats-user', password='pass')
        task = Task.objects.create(
            user=user,
            name='image stats',
            task_type='image-classification',
            data_file='uploads/data/images.zip',
            uploaded_total=8,
            label_list=['a', 'b'],
        )

        ImageResult.objects.bulk_create([
            ImageResult(task=task, sequence_number=1, image_name='m1.jpg', image_path='m1.jpg', label='a', confidence=-1, status='unverified'),
            ImageResult(task=task, sequence_number=2, image_name='m2.jpg', image_path='m2.jpg', label='b', confidence=-1, status='unverified'),
            ImageResult(task=task, sequence_number=3, image_name='m3.jpg', image_path='m3.jpg', label='a', confidence=-1, status='verified'),
            ImageResult(task=task, sequence_number=4, image_name='u1.jpg', image_path='u1.jpg', label='a', confidence=0.61, status='unverified'),
            ImageResult(task=task, sequence_number=5, image_name='u2.jpg', image_path='u2.jpg', label='a', confidence=0.62, status='unverified'),
            ImageResult(task=task, sequence_number=6, image_name='u3.jpg', image_path='u3.jpg', label='b', confidence=0.63, status='unverified'),
            ImageResult(task=task, sequence_number=7, image_name='u4.jpg', image_path='u4.jpg', label='b', confidence=0.64, status='unverified'),
            ImageResult(task=task, sequence_number=8, image_name='u5.jpg', image_path='u5.jpg', label='a', confidence=0.65, status='unverified'),
        ])

        stats = _get_label_verification_stats(task, ImageResult, 'task')

        self.assertEqual(stats['manual_annotated_count'], 3)
        self.assertEqual(stats['unverified_count'], 2)
        self.assertEqual(stats['verified_count'], 1)
        self.assertEqual(stats['unlabeled_count'], 5)
        self.assertEqual(stats['total_verifiable_count'], 3)
        self.assertEqual(stats['verification_progress'], 33.3)
