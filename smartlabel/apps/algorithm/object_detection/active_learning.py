def detection_uncertainty(annotations):
    """Return an image-level active-learning score in [0, 1].

    The score combines the strongest box confidence, average inverse
    confidence, and the ratio of low-confidence boxes. Images with no
    detections receive the maximum score so they are reviewed early.
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

    score = (
        0.55 * (1.0 - max_confidence)
        + 0.35 * mean_inverse_confidence
        + 0.10 * low_confidence_ratio
    )
    return round(max(0.0, min(1.0, score)), 6), low_confidence_count
