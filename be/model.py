import sys
import warnings

import numpy as np
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

sys.path.append("/opt/ml/input/Mask2Former")
from mask2former import add_maskformer2_config

from utils import label_to_color_image

warnings.filterwarnings('ignore')
CONFIG_DIR = "/opt/ml/input/Mask2Former/mask2former_c11_large_fix_all/config.yaml"
WEIGHT_DIR = "/opt/ml/input/Mask2Former/mask2former_c11_large_fix_all/model_0079999.pth"


def setup_cfg():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(CONFIG_DIR)
    cfg.MODEL.WEIGHTS = WEIGHT_DIR
    cfg.freeze()
    return cfg


def get_model() -> DefaultPredictor:
    cfg = setup_cfg()
    return DefaultPredictor(cfg)


def predict_image(model: DefaultPredictor, image: Image) -> np.array:
    image = np.asarray(image)
    print(image.shape)
    pred = model(image)
    output = pred['sem_seg'].argmax(dim=0).detach().cpu().numpy()
    output = label_to_color_image(output)

    return output
