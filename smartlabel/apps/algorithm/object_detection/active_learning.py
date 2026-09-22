import math

from PIL import Image


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def image_quality_score(image_path):
    """Return a lightweight image complexity/quality score in [0, 1]."""
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
            edge_total += 1.0 if gradient > 30 else 0.0
            sharp_total += gradient / 510.0
            edge_count += 1

    edge_density = edge_total / edge_count if edge_count else 0.0
    sharpness = sharp_total / edge_count if edge_count else 0.0
    return round(_clamp01(0.45 * entropy + 0.30 * edge_density + 0.25 * sharpness), 6)


def image_feature_vector(image_path, size=16):
    """Extract a small normalized grayscale feature vector for similarity ranking."""
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
    if not left or not right or len(left) != len(right):
        return 0.0
    return _clamp01((sum(a * b for a, b in zip(left, right)) + 1.0) / 2.0)


def build_detection_review_context(image_by_name, labeled_map, class_list):
    """Build the closed-loop state used by detection review sample selection."""
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
    if not feature or not labeled_features:
        return 0.5
    nearest = max(cosine_similarity(feature, labeled) for labeled in labeled_features)
    return _clamp01(1.0 - nearest)


def detection_uncertainty(annotations, pre_filter_count=None, post_filter_count=None):
    """Return an image-level boxed-detection uncertainty score in [0, 1]."""
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
    miss_fix_count = int(context.get("miss_fix_count") or 0)
    class_counts = context.get("class_counts") or {}
    queue_repeat_ratio = float(context.get("queue_repeat_ratio") or 0.0)
    positive_count = len(context.get("positive_features") or [])

    if branch_type == "final_empty":
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
        weights["class_balance"] += 0.05
        weights["uncertainty"] -= 0.03
        weights["diversity"] -= 0.02
        weight_version += 10

    if queue_repeat_ratio > 0.35:
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
    """Calculate the patent-oriented closed-loop review value and score details."""
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
    review_value, low_confidence_count, _detail = detection_review_value_detail(
        annotations,
        image_path,
        review_context=review_context,
        pre_filter_count=pre_filter_count,
        post_filter_count=post_filter_count,
    )
    return review_value, low_confidence_count
