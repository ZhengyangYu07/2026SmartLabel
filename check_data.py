from smartlabel.apps.data_management.models import ImageResult

task_id = 119
img_results = ImageResult.objects.filter(task_id=task_id)
print('总数:', img_results.count())
print('status=unverified:', img_results.filter(status='unverified').count())
print('status=unverified AND confidence=1:', img_results.filter(status='unverified', confidence=1).count())
print('status=unverified AND confidence=1.0:', img_results.filter(status='unverified', confidence=1.0).count())

print('\nstatus=unverified 的 confidence 分布:')
for conf in img_results.filter(status='unverified').values_list('confidence', flat=True).distinct().order_by('confidence'):
    cnt = img_results.filter(status='unverified', confidence=conf).count()
    print(f'  {conf}: {cnt} 条')

print('\n所有 status 分布:')
for status in img_results.values_list('status', flat=True).distinct():
    cnt = img_results.filter(status=status).count()
    print(f'  status={status}: {cnt} 条')
