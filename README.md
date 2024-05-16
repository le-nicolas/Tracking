# Basic but semi-advance object detection
I hardly believe nowadays, how advance we are now. Just by less than ~50 lines of code, you can generate a respectable confidence interaval.

![000000000009](https://github.com/le-nicolas/Basic-but-semi-advance-object-detection/assets/112614851/8d952c34-63e6-4f81-b02e-404e14020982)



You may want to prepare for pytorch(nightly), ultralytics and numpy.

just as all machine learning stuff. what we do is it involves training models on data to make predictions or decisions. or to be precise; it involves preprocessing data, training a model, and validating its performance.

the process of this is we load the model for pre-train stuff.
ask it to predict something for us(good thing is it doesnt have to be a single image. you can process bunch of stuff in a single folder.), with a threshold of 0.25

we display tensor array for debugging and understanding the data.(tracking its progress is also in here.)

the crucial part here is, we convert the tensor to numpy array. AND save numpy array to a file.

and there you have it. obj. detection in ~50 lines of code.

![000000000025](https://github.com/le-nicolas/Basic-but-semi-advance-object-detection/assets/112614851/14eeacd1-4300-4a0c-a1d4-829f3708eddc)
![000000000030](https://github.com/le-nicolas/Basic-but-semi-advance-object-detection/assets/112614851/624b6f79-f43f-4f7e-a51c-efcc32bf0010)
![000000000034](https://github.com/le-nicolas/Basic-but-semi-advance-object-detection/assets/112614851/ee250850-6658-4cd0-ada3-636492d6e625)
