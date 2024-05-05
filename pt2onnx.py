import ultralytics

ultralytics.YOLO("./model.pt").export(format="onnx")
