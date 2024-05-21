import ultralytics

ultralytics.YOLO("./model.pt").export(format="onnx", simplify=True, imgsz=(480, 640))
