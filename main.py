#load a pretrained model
model = YOLO('yolov8n.pt', "v8")

#predict on an image with coco8 dataset
detection_output = model.predict(source="datasets/coco8/imagesz/000000000025.jpg", conf=0.25, save=True)

#display tensor array
print(detection_output)

#convert tensor to numpy array
numpy_output = detection_output[0].cpu().numpy() # this returns a copy of the results object with all the tensors as numpy arrays
print(type(detection_output[0])) # The type() function in Python returns the data type of the specified object.
print(detection_output[0].shape) # shape attribute is typically used with numpy arrays or PyTorch tensors to get the dimensions of the array or tensor.
#ensure that your data has the correct number of dimensions and size in each dimension.

#display numpy array
print(numpy_output) # some summary statistics.

# Save numpy array to a file
np.save('output.npy', numpy_output) # never forgetti!
