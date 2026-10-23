from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

def process_video(video_path, output_dir="outputs", confidence=0.35,
                  line_position_percent=65, model_name="yolo26n.pt"):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    evidence_dir = output_dir / "evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video. Try another MP4 file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video frame rate or dimensions.")

    line_y = int(height * (line_position_percent / 100))
    video_out = output_dir / "tracked_traffic.mp4"
    det_out = output_dir / "vehicle_detections.csv"
    events_out = output_dir / "line_crossing_events.csv"
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video. Try an MP4/H.264 input.")

    last_side, crossed, unique_ids = {}, set(), set()
    det_count = event_count = frame_no = 0
    try:
        with det_out.open("w", newline="", encoding="utf-8") as df, events_out.open("w", newline="", encoding="utf-8") as ef:
            dw, ew = csv.writer(df), csv.writer(ef)
            dw.writerow(["frame","time_seconds","track_id","vehicle_type","confidence","x1","y1","x2","y2","center_x","center_y"])
            ew.writerow(["frame","time_seconds","track_id","vehicle_type","event","center_x","center_y","evidence_image"])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_no += 1
                result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                     classes=VEHICLE_CLASSES, conf=float(confidence), verbose=False)[0]
                cv2.line(frame, (0, line_y), (width, line_y), (0, 200, 255), 2)
                cv2.putText(frame, "VIRTUAL REVIEW LINE", (12, max(24, line_y-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
                boxes = result.boxes
                if boxes is not None and len(boxes):
                    coords = boxes.xyxy.cpu().numpy()
                    classes = boxes.cls.int().cpu().tolist()
                    confs = boxes.conf.cpu().tolist()
                    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(classes)
                    for box, cls_id, conf, tid in zip(coords, classes, confs, ids):
                        x1,y1,x2,y2 = [int(v) for v in box]
                        cx,cy = int((x1+x2)/2), int((y1+y2)/2)
                        name = model.names[cls_id]
                        if tid is not None:
                            tid = int(tid)
                            unique_ids.add(tid)
                        dw.writerow([frame_no, round(frame_no/fps,3), "" if tid is None else tid,
                                     name, round(float(conf),4), x1,y1,x2,y2,cx,cy])
                        det_count += 1
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(60,220,60),2)
                        cv2.putText(frame,f"{name} ID:{tid if tid is not None else '-'} {conf:.2f}",
                                    (x1,max(20,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(60,220,60),2)
                        if tid is not None:
                            side = "above" if cy < line_y else "below"
                            old = last_side.get(tid)
                            if old is not None and side != old and tid not in crossed:
                                crossed.add(tid)
                                event_count += 1
                                snap = evidence_dir / f"event_{event_count:04d}_track_{tid}.jpg"
                                cv2.imwrite(str(snap), frame)
                                ew.writerow([frame_no,round(frame_no/fps,3),tid,name,
                                             "virtual_line_crossing_for_review",cx,cy,str(snap)])
                            last_side[tid] = side
                writer.write(frame)
    finally:
        cap.release()
        writer.release()
    return {"video":str(video_out),"detections_csv":str(det_out),"events_csv":str(events_out),
            "evidence_dir":str(evidence_dir),"frames":frame_no,"unique_tracks":len(unique_ids),
            "detections":det_count,"line_crossing_events":event_count,"fps":fps,"width":width,"height":height}
