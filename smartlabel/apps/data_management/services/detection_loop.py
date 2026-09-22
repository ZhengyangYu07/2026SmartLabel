from django.db import transaction
from django.db.models import Max

from smartlabel.apps.algorithm.object_detection.active_learning import image_feature_vector

from ..models import MissFixTable, PositiveIndex, ReviewQueueCache


def _sample_id(task_id, image_name):
    return f"task-{task_id}:{image_name}"


def _annotation_label(annotation):
    label = annotation.get('label') if isinstance(annotation, dict) else ''
    return str(label or '').strip()


def _is_positive_annotation(annotation):
    if not isinstance(annotation, dict):
        return False
    if not _annotation_label(annotation):
        return False
    if annotation.get('decision') == 'rejected':
        return False
    shape_type = annotation.get('shape_type') or 'rectangle'
    if shape_type == 'rectangle':
        return float(annotation.get('w') or 0) > 0 and float(annotation.get('h') or 0) > 0
    return bool(annotation.get('points'))


def _is_manual_annotation(annotation):
    if not _is_positive_annotation(annotation):
        return False
    source = annotation.get('source')
    decision = annotation.get('decision')
    return source == 'manual' or decision in {'manual', 'modified', 'accepted'}


def _positive_labels(annotations, manual_only=False):
    labels = []
    for annotation in annotations or []:
        if manual_only and not _is_manual_annotation(annotation):
            continue
        if not manual_only and not _is_positive_annotation(annotation):
            continue
        label = _annotation_label(annotation)
        if label and label not in labels:
            labels.append(label)
    return labels


def _next_index_version(task):
    current = (
        PositiveIndex.objects
        .filter(task=task)
        .aggregate(max_version=Max('index_version'))
        .get('max_version')
    )
    return int(current or 0) + 1


def get_current_index_version(task):
    current = (
        PositiveIndex.objects
        .filter(task=task, is_active=True)
        .aggregate(max_version=Max('index_version'))
        .get('max_version')
    )
    return int(current or 0)


def get_next_queue_version(task):
    current = (
        ReviewQueueCache.objects
        .filter(task=task)
        .aggregate(max_version=Max('queue_version'))
        .get('max_version')
    )
    return int(current or 0) + 1


def merge_positive_index_context(task, base_context):
    context = dict(base_context or {})
    entries = list(PositiveIndex.objects.filter(task=task, is_active=True))
    if not entries:
        context.setdefault('index_version', 0)
        context.setdefault('miss_fix_count', 0)
        context.setdefault('queue_repeat_ratio', 0.0)
        return context

    class_counts = dict(context.get('class_counts') or {})
    positive_features = []
    labeled_features = list(context.get('labeled_features') or [])
    max_version = 0
    miss_fix_count = 0

    for entry in entries:
        feature = entry.feature_vector or []
        labels = [entry.class_label] if entry.class_label else []
        max_version = max(max_version, int(entry.index_version or 0))
        if entry.source_type == 'miss_fix':
            miss_fix_count += 1
        if entry.class_label:
            class_counts[entry.class_label] = class_counts.get(entry.class_label, 0) + 1
        if feature:
            labeled_features.append(feature)
            positive_features.append({
                'feature': feature,
                'labels': labels,
                'source_type': entry.source_type,
                'index_version': entry.index_version,
            })

    context['class_counts'] = class_counts
    context['positive_features'] = positive_features
    context['labeled_features'] = labeled_features
    context['index_version'] = max_version
    context['miss_fix_count'] = miss_fix_count
    context.setdefault('queue_repeat_ratio', 0.0)
    return context


def apply_detection_loop_update(task, item, previous_annotations, normalized_annotations, operator=None, queue_version=None):
    previous_count = sum(1 for annotation in previous_annotations or [] if _is_positive_annotation(annotation))
    labels = _positive_labels(normalized_annotations, manual_only=True)
    if not labels:
        return None

    source_type = 'miss_fix' if previous_count == 0 else 'manual_positive'
    feature = image_feature_vector(item.image_path)
    sample_id = _sample_id(task.id, item.image_name)
    next_version = _next_index_version(task)
    miss_fix_event = None

    with transaction.atomic():
        if source_type == 'miss_fix':
            miss_fix_event = MissFixTable.objects.create(
                task=task,
                image_result=item,
                sample_id=sample_id,
                manual_bbox=[
                    annotation for annotation in normalized_annotations or []
                    if _is_manual_annotation(annotation)
                ],
                manual_class=labels[0],
                operator=operator,
                pre_fix_box_count=previous_count,
                source_queue_version=queue_version or 1,
                index_write_status='pending',
                positive_index_version=next_version,
            )

        for label in labels:
            PositiveIndex.objects.update_or_create(
                task=task,
                image_result=item,
                class_label=label,
                source_type=source_type,
                defaults={
                    'sample_id': sample_id,
                    'image_name': item.image_name,
                    'feature_vector': feature,
                    'index_version': next_version,
                    'is_active': True,
                },
            )

        if miss_fix_event:
            miss_fix_event.index_write_status = 'written'
            miss_fix_event.save(update_fields=['index_write_status'])
            ReviewQueueCache.objects.filter(
                task=task,
                branch_type='final_empty',
                index_version_snapshot__lt=next_version,
            ).update(cache_status='needs_recompute')

    return {
        'source_type': source_type,
        'index_version': next_version,
        'labels': labels,
        'miss_fix_event_id': miss_fix_event.id if miss_fix_event else None,
    }


def persist_review_queue_items(task, rows, queue_version, progress_callback=None):
    result_ids = {
        item.image_name: item
        for item in task.results.filter(image_name__in=[row.get('image_name') for row in rows])
    }
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        detail = row.get('review_detail') or {}
        item = result_ids.get(row.get('image_name'))
        if not item:
            continue
        branch_type = detail.get('branch_type') or ('boxed' if row.get('annotations') else 'final_empty')
        ReviewQueueCache.objects.update_or_create(
            task=task,
            image_result=item,
            queue_version=queue_version,
            defaults={
                'sample_id': _sample_id(task.id, item.image_name),
                'branch_type': branch_type,
                'review_score': row.get('uncertainty_score') or 0.0,
                'candidate_chaos': detail.get('candidate_chaos') or 0.0,
                'positive_similarity': detail.get('positive_similarity') or 0.0,
                'pre_filter_response': detail.get('pre_filter_response') or 0.0,
                'image_quality': detail.get('image_quality') or 0.0,
                'class_balance': detail.get('class_balance') or 0.0,
                'diversity': detail.get('diversity') or 0.0,
                'weight_version': detail.get('weight_version') or 1,
                'index_version_snapshot': detail.get('index_version_snapshot') or 0,
                'trigger_reason': detail.get('trigger_reason') or '',
                'cache_status': 'fresh',
                'score_detail': detail,
            },
        )
        if progress_callback:
            progress_callback(index, total)
