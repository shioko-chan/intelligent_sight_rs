from ultralytics import YOLO

model = YOLO("./model.pt")

print(model.output_shape)
