# vitg.network.backbone.vitgyolov8 YOLO 🚀, GPL-3.0 license

__version__ = "8.0.38"

from vitg.network.backbone.vitgyolov8.yolo.engine.model import YOLO
from vitg.network.backbone.vitgyolov8.yolo.utils.checks import check_yolo as checks

__all__ = ["__version__", "YOLO", "checks"]  # allow simpler import
