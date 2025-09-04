# unsup_yolo_tiny.py
import cv2
import numpy as np
import time
import argparse

import tensorflow as tf

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov4-tiny.weights")
    p.add_argument("--cfg", default="yolov4-tiny.cfg")
    p.add_argument("--names", default="coco.names")
    p.add_argument("--source", default=0, help="video source: 0 for webcam or path to video")
    p.add_argument("--input-size", type=int, default=416, help="network input size (320 or 416)")
    p.add_argument("--conf-thresh", type=float, default=0.4)
    p.add_argument("--iou-thresh", type=float, default=0.45)
    p.add_argument("--min-area", type=int, default=900, help="min contour area for FG blobs")
    return p.parse_args()


# loading yolo- tensorflow
def load_yolo_tf(model_path):
    model = tf.saved_model.load(model_path)
    infer = model.signatures['serving_default']
    return infer



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

    
    infer = load_yolo_tf(args.weights)
    print(" YoloV4-tiny Tensorflow model reloaded.")

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)     # Video source
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    input_size = args.input_size
    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    min_area = args.min_area

    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]

        # --- YOLO inference ---
        img_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_in = cv2.resize(img_in, (input_size, input_size))
        img_in = img_in / 255.0
        img_in = img_in.astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)        
        
        batch_data = tf.constant(img_in)
        pred = infer(batch_data)  # dict of { 'tf_op_layer_concat': boxes, 'tf_op_layer_conf': scores }

        
        # NOTE: keys depend on model export; common: 'tf_op_layer_concat', 'tf_op_layer_concat_1'
        boxes = pred['tf_op_layer_concat'].numpy()    # shape: (1, N, 4)
        confs = pred['tf_op_layer_concat_1'].numpy()  # shape: (1, N, num_classes)

        boxes, confs = boxes[0], confs[0]

        boxes_scaled, confidences, classIDs = [], [], []
        
        for i in range(len(boxes)):
            box = boxes[i]
            score = confs[i]
            classID = int(np.argmax(score))
            confidence = score[classID]
            if confidence > conf_thresh:
                # YOLO boxes are in normalized format (ymin, xmin, ymax, xmax)
                ymin, xmin, ymax, xmax = box
                x = int(xmin * W)
                y = int(ymin * H)
                w = int((xmax - xmin) * W)
               
                h = int((ymax - ymin) * H)
                boxes_scaled.append([x, y, w, h])
                confidences.append(float(confidence))
                classIDs.append(classID)
        
        idxs = tf.image.non_max_suppression(
            boxes=np.array([ [b[1], b[0], b[1]+b[3], b[0]+b[2]] for b in boxes_scaled ]),
            scores=np.array(confidences),
            max_output_size=50,
            iou_threshold=iou_thresh,
            score_threshold=conf_thresh
        ).numpy()
        
        yolo_detected_boxes=[boxes_scaled[i] for i in idxs]
        
        #drawing yolo detection 
        
        for i in idxs:
            x, y, w, h = boxes_scaled[i]
            label = f"{classes[classIDs[i]]}:{confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
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

        cv2.imshow("unsup-yolo-tf", frame)
        #  also show fgmask for debugging (optional)
        cv2.imshow("fgmask", fgmask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
