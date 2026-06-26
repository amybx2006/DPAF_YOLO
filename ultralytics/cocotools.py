from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
import os

anno_json="I:/UAV/valid/label_json/UAV_val.json" #gt的json文件地址
pred_json="I:/UAV/valid/label_json/UAV_val20251.json" #pred的json文件地址
img_path="I:/UAV/valid/images1" #对应的图像地址
anno = COCO(anno_json)  # init annotations api
pred = anno.loadRes(pred_json)  # init predictions api
eval = COCOeval(anno, pred, 'bbox')

eval.params.imgIds = [int(Path(x).stem) for x in os.listdir(img_path)]  # image IDs to evaluate
eval.evaluate()
eval.accumulate()
eval.summarize()
map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
small_map50 = eval.stats[3]  # 根据 COCO 的标准，小目标的 mAP@0.5 通常在第 3 个位置
print(f"小目标 mAP@0.5: {small_map50:.3f}")

