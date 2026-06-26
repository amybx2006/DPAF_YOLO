from ultralytics import YOLO

if __name__ == '__main__':
    # 直接使用预训练模型创建模型.
    # model = YOLO('yolov8n.pt')
    # model.train(**{'cfg':'ultralytics/cfg/exp1.yaml', 'data':'dataset/data.yaml'})

    # 使用yaml配置文件来创建模型,并导入预训练权重.
    model = YOLO('E:/learn/ultralytics/ultralytics/cfg/models/v5/yolov5s.yaml')
    # model.load('E:/learn/ultralytics/yolo11n.pt')
    model.train(**{'cfg': 'E:/learn/ultralytics/ultralytics/cfg/train.yaml', 'data': 'E:/learn/ultralytics/ultralytics/cfg/datasets/AIRUI_HANGPAI.yaml'})

    # 模型验证
    # model = YOLO('runs/detect/train11/weights/best.pt')
    # model.val(**{'cfg':'ultralytics/cfg/PCB.yaml', 'data':'datasets/VOCPCB.yaml'})

    # 模型推理
    # model = YOLO('runs/detect/yolov8n_exp/best.pt')
    # model.predict(source='dataset/images/test', **{'save':True})

    # 模型导出
    # model = YOLO("Weight/yolov8n.pt")  # load an official model
    # model.export(format="onnx")
# import warnings
#
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     model = YOLO('E:/learn/ultralytics/ultralytics/cfg/models/11/yolo11s.yaml')
#     # 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,
#     # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
#     # model.load('yolov11n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
#     model.train(data=r"E:\learn\ultralytics\ultralytics\cfg\datasets\UAV.yaml",
#                 # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
#                 cache=False,
#                 imgsz=640,
#                 epochs=100,
#                 single_cls=False,  # 是否是单类别检测
#                 batch=4,
#                 close_mosaic=0,
#                 workers=0,
#                 device='0',
#                 optimizer='SGD',  # using SGD 优化器 默认为auto建议大家使用固定的.
#                 # resume=, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
#                 amp=True,  # 如果出现训练损失为Nan可以关闭amp
#                 project='runs/train',
#                 name='exp',
#                 )
# E:\learn\ultralytics\ultralytics\cfg\models\11\yolo11s_add one layer.yaml