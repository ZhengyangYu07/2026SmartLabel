"""
目标检测主动学习（复核价值评分）核心算法。

思路：对每张图计算“复核价值”综合分，分越高越值得人工看。分数由三部分加权：
- uncertainty    不确定性：模型置信度越低/候选框越混乱，越值得复核；
- class_balance  类别均衡：越稀缺的类别得分越高，缓解长尾；
- diversity      多样性：与已标注样本越不相似，越值得补充。

权重由 schedule_detection_review_weights 依据闭环状态（漏检次数、类别是否偏斜、
队列重复率）动态调整。图像特征用 16x16 灰度归一化向量表示，相似度用余弦。
"""
import math

from PIL import Image


def _clamp01(value):
    """把数值裁剪到 [0, 1]。"""
    return max(0.0, min(1.0, float(value)))


def image_quality_score(image_path):
    """Return a lightweight image complexity/quality score in [0, 1].

    用 64x64 灰度图的熵、边缘密度、清晰度三项加权，衡量图像“信息量”。
    打不开图片时退化为按文件大小估算，避免因坏图中断整批处理。
    """
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L").resize((64, 64))
            pixels = list(gray.getdata())
    except Exception:
        try:
            return _clamp01(image_path.stat().st_size / (1024 * 1024))
        except Exception:
            return 0.0

    if not pixels:
        return 0.0

    histogram = [0] * 256
    for value in pixels:
        histogram[value] += 1
    total = float(len(pixels))
    # 归一化信息熵（除以 8 是因为灰度最多 8 bit）
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in histogram
        if count
    ) / 8.0

    edge_total = 0.0
    sharp_total = 0.0
    edge_count = 0
    width = 64
    for y in range(width - 1):
        row = y * width
        next_row = (y + 1) * width
        for x in range(width - 1):
            dx = abs(pixels[row + x] - pixels[row + x + 1])
            dy = abs(pixels[row + x] - pixels[next_row + x])
            gradient = dx + dy
            edge_total += 1.0 if gradient > 30 else 0.0  # 梯度超阈值计为边缘
            sharp_total += gradient / 510.0
            edge_count += 1

    edge_density = edge_total / edge_count if edge_count else 0.0
    sharpness = sharp_total / edge_count if edge_count else 0.0
    return round(_clamp01(0.45 * entropy + 0.30 * edge_density + 0.25 * sharpness), 6)


def image_feature_vector(image_path, size=16):
    """Extract a small normalized grayscale feature vector for similarity ranking.

    返回 16x16=256 维、去均值后按 L2 归一化的向量，便于直接做余弦相似度。
    读取失败返回空列表，调用方需容错。
    """
    try:
        with Image.open(image_path) as img:
            pixels = [float(v) / 255.0 for v in img.convert("L").resize((size, size)).getdata()]
    except Exception:
        return []

    if not pixels:
        return []

    mean_value = sum(pixels) / len(pixels)
    centered = [value - mean_value for value in pixels]
    norm = math.sqrt(sum(value * value for value in centered)) or 1.0
    return [value / norm for value in centered]


def cosine_similarity(left, right):
    """余弦相似度映射到 [0,1]（-1→0, 1→1）；维度不符或空向量返回 0。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    return _clamp01((sum(a * b for a, b in zip(left, right)) + 1.0) / 2.0)


def build_detection_review_context(image_by_name, labeled_map, class_list):
    """Build the closed-loop state used by detection review sample selection.

    基于“已标注样本”初始化闭环上下文：统计类别计数、提取正样本与已标注特征。
    后续 merge_positive_index_context 会把人工确认的正样本索引合并进来。
    """
    class_counts = {label: 0 for label in class_list or []}
    positive_features = []
    labeled_features = []

    for image_name, annotations in (labeled_map or {}).items():
        image_path = image_by_name.get(image_name)
        if not image_path:
            continue

        labels = []
        for annotation in annotations or []:
            label = (annotation.get("label") or "").strip()
            if not label:
                continue
            class_counts[label] = class_counts.get(label, 0) + 1
            labels.append(label)

        feature = image_feature_vector(image_path)
        if feature:
            labeled_features.append(feature)
            if labels:
                positive_features.append({"feature": feature, "labels": labels})

    return {
        "class_counts": class_counts,
        "positive_features": positive_features,
        "labeled_features": labeled_features,
    }


def _class_balance_score(labels, class_counts):
    """类别均衡分：越稀少的类别得分越高（按平均计数/该类计数的平方根）。"""
    if not labels or not class_counts:
        return 0.5
    avg_count = sum(class_counts.values()) / float(len(class_counts) or 1)
    scores = []
    for label in labels:
        count = class_counts.get(label, 0)
        scores.append(math.sqrt((avg_count + 1.0) / (count + 1.0)))
    if not scores:
        return 0.5
    return _clamp01(max(scores) / 2.0)


def _nearest_positive(feature, positive_features):
    """找到与当前特征最相似的正样本，返回 (相似度, 其类别, 其来源类型)。"""
    best_score = 0.0
    best_labels = []
    best_source_type = ""
    for item in positive_features or []:
        score = cosine_similarity(feature, item.get("feature") or [])
        if score > best_score:
            best_score = score
            best_labels = item.get("labels") or []
            best_source_type = item.get("source_type") or ""
    return best_score, best_labels, best_source_type


def _diversity_score(feature, labeled_features):
    """多样性分：与最近的已标注样本越不相似，得分越高（=1-最近相似度）。"""
    if not feature or not labeled_features:
        return 0.5
    nearest = max(cosine_similarity(feature, labeled) for labeled in labeled_features)
    return _clamp01(1.0 - nearest)


def detection_uncertainty(annotations, pre_filter_count=None, post_filter_count=None):
    """Return an image-level boxed-detection uncertainty score in [0, 1].

    综合：最高置信度越低、平均不确定度越高、低置信框占比越大、候选过滤比例越异常，
    不确定性越高。返回 (分数, 低置信框数量)。
    """
    if not annotations:
        return 1.0, 0

    confidences = []
    for annotation in annotations:
        try:
            confidence = float(annotation.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidences.append(max(0.0, min(1.0, confidence)))

    if not confidences:
        return 1.0, 0

    max_confidence = max(confidences)
    mean_inverse_confidence = sum(1.0 - value for value in confidences) / len(confidences)
    low_confidence_count = sum(1 for value in confidences if value < 0.5)
    low_confidence_ratio = low_confidence_count / len(confidences)
    # 过滤前后候选数之比越大，说明模型候选越“发散”，越混乱
    pre_count = max(float(pre_filter_count or len(confidences)), 0.0)
    post_count = max(float(post_filter_count or len(confidences)), 0.0)
    candidate_ratio = pre_count / (post_count + 1.0)
    conflict_score = math.log1p(candidate_ratio) / math.log1p(10.0)

    score = (
        0.45 * (1.0 - max_confidence)
        + 0.25 * mean_inverse_confidence
        + 0.15 * low_confidence_ratio
        + 0.15 * _clamp01(conflict_score)
    )
    return round(max(0.0, min(1.0, score)), 6), low_confidence_count


def schedule_detection_review_weights(context, branch_type):
    """根据闭环状态动态计算三维权重并归一化。

    依据：漏检补框次数、是否类别偏斜、队列重复率；weight_version 记录权重档位，
    便于追溯与对比（+10 表示类别偏斜调整，+20 表示重复率调整）。
    """
    miss_fix_count = int(context.get("miss_fix_count") or 0)
    class_counts = context.get("class_counts") or {}
    queue_repeat_ratio = float(context.get("queue_repeat_ratio") or 0.0)
    positive_count = len(context.get("positive_features") or [])

    if branch_type == "final_empty":
        # 无框分支：样本少或漏检反馈不足时，先靠不确定性主导
        if positive_count == 0 or miss_fix_count < 3:
            weights = {
                "uncertainty": 0.62,
                "class_balance": 0.18,
                "diversity": 0.20,
            }
            weight_version = 1
        else:
            weights = {
                "uncertainty": 0.74,
                "class_balance": 0.16,
                "diversity": 0.10,
            }
            weight_version = 2
    else:
        weights = {
            "uncertainty": 0.70,
            "class_balance": 0.20,
            "diversity": 0.10,
        }
        weight_version = 1

    counts = list(class_counts.values())
    if counts and min(counts) <= 1:
        # 存在样本数 ≤1 的稀有类别 → 提高类别均衡权重
        weights["class_balance"] += 0.05
        weights["uncertainty"] -= 0.03
        weights["diversity"] -= 0.02
        weight_version += 10

    if queue_repeat_ratio > 0.35:
        # 候选队列重复率高 → 提高多样性权重，避免反复挑同一批样本
        weights["diversity"] += 0.08
        weights["uncertainty"] -= 0.05
        weights["class_balance"] -= 0.03
        weight_version += 20

    total = sum(max(value, 0.0) for value in weights.values()) or 1.0
    return {key: _clamp01(value / total) for key, value in weights.items()}, weight_version


def detection_review_value_detail(
    annotations,
    image_path,
    review_context=None,
    pre_filter_count=0,
    post_filter_count=0,
):
    """Calculate the patent-oriented closed-loop review value and score details.

    核心入口：返回 (复核价值分, 低置信框数, 评分明细)。
    有框图走不确定性分支，无框图走“正样本相似度”分支，再与类别均衡、多样性加权。
    """
    context = review_context or {}
    class_counts = context.get("class_counts") or {}
    feature = image_feature_vector(image_path)
    quality = image_quality_score(image_path)
    branch_type = "boxed" if annotations else "final_empty"
    positive_similarity = 0.0
    positive_source_type = ""
    candidate_chaos = 0.0
    pre_filter_response = _clamp01(float(pre_filter_count or 0) / 10.0)

    if annotations:
        uncertainty, low_confidence_count = detection_uncertainty(
            annotations,
            pre_filter_count=pre_filter_count,
            post_filter_count=post_filter_count or len(annotations),
        )
        labels = [(annotation.get("label") or "").strip() for annotation in annotations]
        pre_count = max(float(pre_filter_count or len(annotations)), 0.0)
        post_count = max(float(post_filter_count or len(annotations)), 0.0)
        candidate_ratio = pre_count / (post_count + 1.0)
        candidate_chaos = _clamp01(math.log1p(candidate_ratio) / math.log1p(10.0))
    else:
        # 无框图：与已知正样本比相似度，越像“已确认的目标”越值得复核
        low_confidence_count = 0
        similarity, labels, positive_source_type = _nearest_positive(
            feature,
            context.get("positive_features") or [],
        )
        if not context.get("positive_features"):
            similarity = quality
        positive_similarity = similarity
        uncertainty = _clamp01(0.45 * similarity + 0.35 * pre_filter_response + 0.20 * quality)

    class_balance = _class_balance_score(labels, class_counts)
    diversity = _diversity_score(feature, context.get("labeled_features") or [])
    weights, weight_version = schedule_detection_review_weights(context, branch_type)
    review_value = (
        weights["uncertainty"] * uncertainty
        + weights["class_balance"] * class_balance
        + weights["diversity"] * diversity
    )
    detail = {
        "branch_type": branch_type,
        "uncertainty": round(_clamp01(uncertainty), 6),
        "candidate_chaos": round(_clamp01(candidate_chaos), 6),
        "positive_similarity": round(_clamp01(positive_similarity), 6),
        "positive_source_type": positive_source_type,
        "pre_filter_response": round(_clamp01(pre_filter_response), 6),
        "image_quality": round(_clamp01(quality), 6),
        "class_balance": round(_clamp01(class_balance), 6),
        "diversity": round(_clamp01(diversity), 6),
        "weights": weights,
        "weight_version": weight_version,
        "index_version_snapshot": int(context.get("index_version") or 0),
        # 记录“为什么挑中它”，用于前端展示与追溯
        "trigger_reason": (
            "miss_fix_positive_similarity"
            if positive_source_type == "miss_fix"
            else "positive_similarity" if branch_type == "final_empty"
            else "candidate_chaos"
        ),
    }
    return round(_clamp01(review_value), 6), low_confidence_count, detail


def detection_review_value(
    annotations,
    image_path,
    review_context=None,
    pre_filter_count=0,
    post_filter_count=0,
):
    """只要分数的简化入口，内部复用 detection_review_value_detail。"""
    review_value, low_confidence_count, _detail = detection_review_value_detail(
        annotations,
        image_path,
        review_context=review_context,
        pre_filter_count=pre_filter_count,
        post_filter_count=post_filter_count,
    )
    return review_value, low_confidence_count
