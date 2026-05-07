from dataclasses import dataclass
from typing import List, Sequence

from utils.data_utils import load_dataset, encode_labels, get_mask, load_dataset_from_dir


@dataclass
class TopicSample:
    text: str
    label: str = ""
    is_labeled: bool = False


def normalize_topic_samples(rows: Sequence[dict]) -> List[TopicSample]:
    samples: List[TopicSample] = []
    for row in rows:
        samples.append(
            TopicSample(
                text=str(row.get("text") or ""),
                label=str(row.get("label") or ""),
                is_labeled=bool(row.get("is_labeled", False)),
            )
        )
    return samples
