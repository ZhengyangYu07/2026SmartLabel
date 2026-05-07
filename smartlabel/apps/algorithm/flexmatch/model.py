from models.CG3_Paper import CG3_Paper
from models.LLGC import llgc_predict
from models.ManifoldClassifier import ManifoldClassifier
from models.MixTextClassifier import MixTextClassifier, mixup_data
from models.losses import contrastive_loss, structure_generation_loss


class TopicClassificationModel:
    def __init__(self, config: dict):
        self.config = config

    def predict(self, texts):
        raise NotImplementedError("主题分类模型推理应由当前平台训练脚本提供。")

