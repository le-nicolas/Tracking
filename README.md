# Tracking

Computer vision project that tracks an object in video and predicts its near-future trajectory from recent motion.

## Features

- Track an object after manual ROI selection (`cv2.selectROI`)
- Real-time path overlay for observed trajectory
- Future-point prediction using polynomial regression on recent motion
- Works with webcam or video file input
- Reset tracking target at runtime (`r`)

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run with webcam:

```bash
python trajectory_tracker.py --source 0
```

4. Or run with a video file:

```bash
python trajectory_tracker.py --source ./sample.mp4
```

## Controls

- `q`: quit
- `r`: reselect the tracked object

## Notes

- Tracking uses CamShift with an HSV histogram model from your selected ROI.
- Prediction is computed from recent center points and updates every frame.
