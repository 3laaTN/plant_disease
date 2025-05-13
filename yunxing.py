
from ultralytics import YOLO 
#训练

# model = YOLO('yolov8n.pt')
model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml').load('yolov8n.pt')# 加载预训练的 YOLOv8n 模型
model.train(data='11000.yaml',epochs=300, batch=32,imgsz=640) # 训练模型8

# model = YOLO('ultralytics/cfg/models/v8/yolov8-seg.yaml').load('yolov8x-seg.pt')# 加载预训练的 YOLOv8n 模型
# model.train(data='11000.yaml',epochs=300, batch=32,imgsz=640) # 训练模型8


#验证 
# 
# model = YOLO('runs/detect/train386/weights/best.pt')
# model.val(data='11000.yaml',split='test',batch=16) # 在验证集模型上评估模型性能3
# model = YOLO('runs/detect/train418/weights/best.pt')
# model.val(data='11000.yaml',split='val',batch=32) # 在验证集模型上评估模型性能


# 测试

# model = YOLO('runs/detect/train386/weights/best.pt')
# model = YOLO('runs/detect/train142/weights/best.pt')
# model.predict(source='datasets/bdz/images/test', device=0, batch=16, save_txt=True,save=True) # 对图像进行预测
# model.predict(source='images', device=0, batch=1, save_txt=True,save=True) # 对图像进行预测
# model.predict(source='datasets/2000/images/test', device=0, batch=16, save_txt=True,save=True) # 对图像进行预测

# model.predict(source='images/video_3.mp4', device=0, save_txt=True,save=True, show=True)
# model.predict(source='datasets/tt100k/images/test', device=0, batch=16, save_txt=True,save=True) # 对图像进行预测


# #导出部署
# model = YOLO('weight/yolov8s-pose.pt')
# model.export(format='onnx',save=True) # 将模型导出为 ONNX 格式