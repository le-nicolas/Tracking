# Basic but semi-advance object detection
I hardly believe nowadays, how advance we are now. Just by less than ~20 lines of code, you can generate a respectable confidence interval.

![000000000009](https://github.com/le-nicolas/Basic-but-semi-advance-object-detection/assets/112614851/8d952c34-63e6-4f81-b02e-404e14020982)



You may want to prepare for pytorch(nightly), ultralytics and numpy.

just as all machine learning stuff. what we do is it involves training models on data to make predictions or decisions. or to be precise; it involves preprocessing data, training a model, and validating its performance.

the process of this is we load the model for pre-train stuff.
ask it to predict something for us(good thing is it doesnt have to be a single image. you can process bunch of stuff in a single folder.), with a threshold of 0.25

we display tensor array for debugging and understanding the data.(tracking its progress is also in here.)

the crucial part here is, we convert the tensor to numpy array. AND save numpy array to a file.

and there you have it. obj. detection in ~20 lines of code. ps. the other girrafe and the trees was not detected. but still im kind of proud of this bad boy.
pps. here are the good stuff, this is how it runs, under the hood
![image](https://github.com/le-nicolas/Basic-but-semi-advance-object-detection/assets/112614851/e0b66b8d-e170-453e-97ac-b7f8922301b1)

names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

![000000000025](https://github.com/le-nicolas/Basic-but-semi-advance-object-detection/assets/112614851/14eeacd1-4300-4a0c-a1d4-829f3708eddc)
![000000000030](https://github.com/le-nicolas/Basic-but-semi-advance-object-detection/assets/112614851/624b6f79-f43f-4f7e-a51c-efcc32bf0010)
![000000000034](https://github.com/le-nicolas/Basic-but-semi-advance-object-detection/assets/112614851/ee250850-6658-4cd0-ada3-636492d6e625)
