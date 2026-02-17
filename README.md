# MediaPipe Node Lab

Small browser prototype for dancer/media artists to test a node-style pipeline:

- webcam input node
- face extraction node (MediaPipe Face Landmarker)
- hand extraction node (MediaPipe Hand Landmarker)
- gesture mapping node
- stage output node with reactive visuals

## Run

1. Open a terminal in this folder:

```bash
cd /Users/taeyang/Developer/mediapipe-node-lab
```

2. Start a local server (required for camera + module loading):

```bash
python3 -m http.server 4173
```

3. Open:

`http://localhost:4173`

4. Click `Start Camera + Models`.

## Notes

- Browser must allow camera permission.
- `localhost` is treated as a secure context, so camera access works.
- Models are loaded from official MediaPipe model storage:
  - face: `face_landmarker.task`
  - hand: `hand_landmarker.task`

## Graph Overview

Default graph created at startup:

`Webcam Source -> Face Extract`
`Webcam Source -> Hand Extract`
`(Webcam + Face + Hand) -> Landmark Overlay`
`(Face metrics + Hand metrics) -> Gesture Mapper -> Stage Output`

This gives a direct "extract then derive" structure that can be expanded with more nodes.

## Next Experiments

- Add a `Mask Generator` node (e.g. face ROI crop).
- Add OSC/WebSocket output node for TouchDesigner/Max.
- Add model switch node (selfie segmentation, pose landmarker).
