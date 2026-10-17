"""
目标检测半监督闭环服务。

负责把“人工标注/复核”的结果回写到三张闭环表：
- 依据人工确认的标注写入 PositiveIndex（正样本索引）；
- 若原本无框、人工补框，则额外记一条 MissFixTable（漏检补框事件）；
- 每次写入使索引版本 +1，并把过期的复核队列缓存标记为待重算。

对外入口：apply_detection_loop_update（人工提交后调用）、persist_review_queue_items（生成候选队列时调用）。
"""
from django.db import transaction
from django.db.models import Max

from smartlabel.apps.algorithm.object_detection.active_learning import image_feature_vector

from ..models import MissFixTable, PositiveIndex, ReviewQueueCache


def _sample_id(task_id, image_name):
    """生成全局唯一的样本标识，与算法侧约定格式一致。"""
    return f"task-{task_id}:{image_name}"


def _annotation_label(annotation):
    """取出标注的类别名，非字典或空值一律返回空串。"""
    label = annotation.get('label') if isinstance(annotation, dict) else ''
    return str(label or '').strip()


def _is_positive_annotation(annotation):
    """判断标注是否为“有效正样本”：有类别、未被驳回、且几何信息有效。"""
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
    """判断标注是否由人工产生（而非模型原始输出）。"""
    if not _is_positive_annotation(annotation):
        return False
    source = annotation.get('source')
    decision = annotation.get('decision')
    return source == 'manual' or decision in {'manual', 'modified', 'accepted'}


def _positive_labels(annotations, manual_only=False):
    """收集标注中的正样本类别，去重并保持出现顺序。"""
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
    """下一个索引版本号 = 当前最大版本 + 1。"""
    current = (
        PositiveIndex.objects
        .filter(task=task)
        .aggregate(max_version=Max('index_version'))
        .get('max_version')
    )
    return int(current or 0) + 1


def get_current_index_version(task):
    """当前生效（is_active）索引的最高版本号，无记录时为 0。"""
    current = (
        PositiveIndex.objects
        .filter(task=task, is_active=True)
        .aggregate(max_version=Max('index_version'))
        .get('max_version')
    )
    return int(current or 0)


def get_next_queue_version(task):
    """下一个复核队列版本号 = 当前最大队列版本 + 1。"""
    current = (
        ReviewQueueCache.objects
        .filter(task=task)
        .aggregate(max_version=Max('queue_version'))
        .get('max_version')
    )
    return int(current or 0) + 1


def merge_positive_index_context(task, base_context):
    """把数据库中的正样本索引合并进算法上下文，供挑样本时使用。

    产出/更新这些键：class_counts（类别计数）、positive_features（正样本特征）、
    labeled_features（已标注特征）、index_version、miss_fix_count。
    """
    context = dict(base_context or {})
    entries = list(PositiveIndex.objects.filter(task=task, is_active=True))
    if not entries:
        # 无索引时给出默认值，保证下游评分函数不会因缺键报错
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
    """人工提交标注后回写闭环状态。

    - previous_annotations 为空（原本无框）而本次有框 → 记为 miss_fix（漏检补框）；
    - 否则记为 manual_positive（人工确认正样本）。
    返回写入摘要；若本次没有有效人工标注则返回 None（不做任何写入）。
    """
    previous_count = sum(1 for annotation in previous_annotations or [] if _is_positive_annotation(annotation))
    labels = _positive_labels(normalized_annotations, manual_only=True)
    if not labels:
        return None

    source_type = 'miss_fix' if previous_count == 0 else 'manual_positive'
    feature = image_feature_vector(item.image_path)
    sample_id = _sample_id(task.id, item.image_name)
    next_version = _next_index_version(task)
    miss_fix_event = None

    # 事件记录与正样本索引必须同事务写入，避免出现“有事件无索引”的不一致状态
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
            # 同一 (任务, 图片, 类别, 来源) 只保留一条，重复提交时覆盖并刷新特征/版本
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
            # 索引已更新，旧的无框候选队列基于过期索引，标记为待重算
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
    """把本次挑出的复核候选写入 ReviewQueueCache。

    rows 为算法侧产出的候选列表（含 image_name、评分与 review_detail）；
    progress_callback(index, total) 用于向任务进度回传。
    """
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
