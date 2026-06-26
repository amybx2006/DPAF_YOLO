# import warnings
#
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     model = YOLO('C:/Users/10595/Desktop/YOLOv11s/yolov11s_chonggou/weights/best.pt')
#     model.predict(source='E:/UAV_V11/ImageSets/val/images',
#                   imgsz=640,
#                   project='runs/detect',
#                   name='exp5',
#                   save=True,
#                   save_txt=False,
#                   conf=0.2,
#                   iou=0.6,
#                   )
# "E:\UAV_V11\ImageSets\val\images"
# C:/Users/10595/Desktop/yolov11s_ppa_dasi_mpdiou/weights/best.pt
# "C:\Users\10595\Desktop\YOLOv11s\YOLOV11\weights\best.pt"


#
# import os
# from ultralytics import YOLO
#
# def save_txt_with_confidence(results, label_dir):
#     """
#     保存预测结果到 TXT 文件，包含置信度
#     :param results: 预测结果
#     :param label_dir: 标签保存目录
#     """
#     os.makedirs(label_dir, exist_ok=True)  # 确保目录存在
#
#     for result in results:
#         # 获取图片文件名（不带扩展名）
#         image_name = os.path.splitext(os.path.basename(result.path))[0]
#         txt_path = os.path.join(label_dir, f"{image_name}.txt")
#
#         # 获取预测结果
#         boxes = result.boxes
#         with open(txt_path, 'w') as f:
#             for box in boxes:
#                 # 获取类别、边界框和置信度
#                 class_id = int(box.cls)  # 类别 ID
#                 xywh = box.xywhn.cpu().numpy()[0]  # 归一化的边界框 (x_center, y_center, width, height)
#                 confidence = box.conf.item()  # 置信度
#
#                 # 写入 TXT 文件：class_id x_center y_center width height confidence
#                 f.write(f"{class_id} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]} {confidence}\n")
#
# def generate_empty_txt_for_no_detections(image_dir, label_dir):
#     """
#     为没有检测到目标的图片生成空的 TXT 文件
#     :param image_dir: 图片文件夹路径
#     :param label_dir: 标签文件夹路径
#     """
#     # 遍历图片文件夹
#     for image_name in os.listdir(image_dir):
#         if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
#             # 对应的 TXT 文件名
#             txt_name = os.path.splitext(image_name)[0] + '.txt'
#             txt_path = os.path.join(label_dir, txt_name)
#
#             # 如果 TXT 文件不存在，创建空文件
#             if not os.path.exists(txt_path):
#                 with open(txt_path, 'w') as f:
#                     pass  # 创建空文件
#
# if __name__ == '__main__':
#     # 加载模型
#     model = YOLO('C:/Users/10595/Desktop/yolov11s_ppa_dasi_mpdiou/weights/best.pt')
#
#     # 运行预测
#     results = model.predict(
#         source='E:/UAV_V11/ImageSets/val/images',  # 输入图像路径
#         imgsz=640,                                # 图像大小
#         project='runs/detect',                    # 结果保存目录
#         name='exp',                               # 结果保存子目录
#         save=False,                               # 不保存预测结果图像
#         save_txt=False,                           # 不自动保存 TXT 文件
#         conf=0.2,                                 # 置信度阈值
#         iou=0.6,                                  # IOU 阈值
#     )
#
#     # 保存预测结果到 TXT 文件（包含置信度）
#     label_dir = 'runs/detect/exp4/labels'
#     save_txt_with_confidence(results, label_dir)
#
#     # 为没有检测到目标的图片生成空的 TXT 文件
#     image_dir = 'E:/UAV_V11/ImageSets/val/images'
#     generate_empty_txt_for_no_detections(image_dir, label_dir)

######################################################################下面的是用于COCOTOOLS的
import os
from ultralytics import YOLO

def save_txt_with_confidence(results, label_dir):
    """
    保存预测结果到 TXT 文件，包含置信度
    :param results: 预测结果
    :param label_dir: 标签保存目录
    """
    os.makedirs(label_dir, exist_ok=True)  # 确保目录存在

    for result in results:
        # 获取图片文件名（不带扩展名）
        image_name = os.path.splitext(os.path.basename(result.path))[0]
        txt_path = os.path.join(label_dir, f"{image_name}.txt")

        # 获取预测结果
        boxes = result.boxes
        with open(txt_path, 'w') as f:
            for box in boxes:
                # 获取类别、边界框和置信度
                class_id = int(box.cls)  # 类别 ID
                xywh = box.xywhn.cpu().numpy()[0]  # 归一化的边界框 (x_center, y_center, width, height)
                confidence = box.conf.item()  # 置信度

                # 写入 TXT 文件：class_id x_center y_center width height confidence
                f.write(f"{class_id} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]} {confidence}\n")

def generate_empty_txt_for_no_detections(image_dir, label_dir):
    """
    为没有检测到目标的图片生成空的 TXT 文件
    :param image_dir: 图片文件夹路径
    :param label_dir: 标签文件夹路径
    """
    # 遍历图片文件夹
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            # 对应的 TXT 文件名
            txt_name = os.path.splitext(image_name)[0] + '.txt'
            txt_path = os.path.join(label_dir, txt_name)

            # 如果 TXT 文件不存在，创建空文件
            if not os.path.exists(txt_path):
                with open(txt_path, 'w') as f:
                    pass  # 创建空文件

if __name__ == '__main__':
    # 加载模型
    model = YOLO('C:/Users/10595/Desktop/yolov11s_ppa_dasi_mpdiou/weights/best.pt')

    # 'C:/Users/10595/Desktop/yolov11s_ppa_dasi_mpdiou/weights/best.pt'
    # 运行预测
    results = model.predict(
        source='I:/UAV/valid/images1',  # 输入图像路径
        imgsz=640,                                # 图像大小
        project='runs/detect',                    # 结果保存目录
        name='exp',                               # 结果保存子目录
        save=False,                               # 不保存预测结果图像
        save_txt=True,                           # 不自动保存 TXT 文件
        conf=0.2,                                 # 置信度阈值
        iou=0.6,                                  # IOU 阈值
    )

    # 保存预测结果到 TXT 文件（包含置信度）
    label_dir = 'runs/detect/exp11/labels'
    save_txt_with_confidence(results, label_dir)

    # 为没有检测到目标的图片生成空的 TXT 文件
    image_dir = 'I:/UAV/valid/images1'
    generate_empty_txt_for_no_detections(image_dir, label_dir)

# "C:\Users\10595\Desktop\YOLOv11s\YOLOV11"