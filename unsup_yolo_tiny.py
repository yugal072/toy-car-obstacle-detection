# unsup_yolo_tiny.py
import cv2
import numpy as np
import time
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov4-tiny.weights")
    p.add_argument("--cfg", default="yolov4-tiny.cfg")
    p.add_argument("--names", default="coco.names")
    p.add_argument("--source", default=0, help="video source: 0 for webcam or path to video")
    p.add_argument("--input-size", type=int, default=416, help="network input size (320 or 416)")
    p.add_argument("--conf-thresh", type=float, default=0.4)
    p.add_argument("--nms-thresh", type=float, default=0.4)
    p.add_argument("--min-area", type=int, default=900, help="min contour area for FG blobs")
    return p.parse_args()

def load_yolo(cfg_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # Use CPU by default; if you have OpenCV built with CUDA, you can set CUDA target
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ln = net.getLayerNames()
    try:
        out_layers = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except:
        out_layers = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, out_layers

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]
    if (boxAArea + boxBArea - interArea) == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def main():
    args = parse_args()

    # load class names
    with open(args.names) as f:
        classes = [c.strip() for c in f.readlines()]

    net, out_names = load_yolo(args.cfg, args.weights)
    print("YOLO loaded. Output layers:", out_names)

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    input_size = args.input_size
    conf_thresh = args.conf_thresh
    nms_thresh = args.nms_thresh
    min_area = args.min_area

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]

        # --- YOLO inference ---
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(out_names)

        boxes = []
        confidences = []
        classIDs = []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classID = int(np.argmax(scores))
                confidence = float(scores[classID])
                if confidence > conf_thresh:
                    cx = int(detection[0] * W)
                    cy = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)
                    x = max(0, int(cx - w/2))
                    y = max(0, int(cy - h/2))
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    classIDs.append(classID)
        # NMS
        idxs = []
        if len(boxes) > 0:
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
            idxs = idxs.flatten() if isinstance(idxs, (list, tuple, np.ndarray)) else [int(x) for x in idxs]

        yolo_detected_boxes = [boxes[i] for i in idxs] if len(idxs) > 0 else []
        # draw YOLO boxes
        for i in idxs:
            x,y,w,h = boxes[i]
            label = f"{classes[classIDs[i]]}:{confidences[i]:.2f}"
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(frame, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)

        # --- Unsupervised: background subtraction (MOG2) ---
        fgmask = fgbg.apply(frame)
        # shadows are 127 if detectShadows=True, remove them by thresholding
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x,y,w,h = cv2.boundingRect(c)
            candidate = [x,y,w,h]
            # check overlap vs YOLO detections. if overlap with a YOLO box, we consider it already handled
            overlaps = [iou(candidate, yb) for yb in yolo_detected_boxes]
            if len(overlaps) > 0 and max(overlaps) > 0.3:
                # it's probably the same object YOLO detected -> skip drawing 'unknown'
                continue
            # else mark as unknown obstacle
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,200,200), 2)
            cv2.putText(frame, "unknown_obstacle", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,200), 2)

        # FPS
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time + 1e-6)
        prev_time = cur_time
        cv2.putText(frame, f"FPS:{fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("unsup-yolo", frame)
        # optional: also show fgmask for debugging
        cv2.imshow("fgmask", fgmask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
