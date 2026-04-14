import argparse
import threading
import time
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class CameraStitcherYOLO:
    def __init__(
        self,
        camera_indices=None,
        debug=False,
        yolo_model='yolov8n.pt',
        person_conf=0.35,
        detect_every=3,
        input_size=(640, 480),
    ):
        if camera_indices is None:
            camera_indices = [0, 1]

        self.camera_indices = camera_indices
        self.debug = debug
        self.width, self.height = input_size

        self.cameras = []
        self.frames = {}
        self.frame_lock = threading.Lock()
        self.running = True

        self.sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.04, edgeThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        self.homographies = []
        self.is_calibrated = False
        self.output_size = None
        self.translation = None

        self.yolo_model_path = yolo_model
        self.person_conf = person_conf
        self.detect_every = max(1, int(detect_every))
        self.detector = None
        self.last_person_boxes = []
        self.last_person_count = 0
        self.frame_count = 0

        self.setup_cameras()
        self.setup_yolo()

    def setup_cameras(self):
        print('Đang khởi tạo cameras...')
        for idx in self.camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                self.cameras.append(cap)
                print(f'Camera {idx} đã được khởi tạo')
            else:
                print(f'Không thể mở camera {idx}')

        if len(self.cameras) < 2:
            raise ValueError('Cần ít nhất 2 camera để thực hiện stitching')

    def setup_yolo(self):
        if YOLO is None:
            print('Ultralytics YOLO chưa được cài. Chạy: pip install ultralytics')
            return
        try:
            self.detector = YOLO(self.yolo_model_path)
            print(f'✓ Đã tải YOLO model: {self.yolo_model_path}')
        except Exception as e:
            print(f'Không thể tải YOLO model: {e}')
            self.detector = None

    def capture_frames(self):
        while self.running:
            current_frames = {}
            for i, cap in enumerate(self.cameras):
                ret, frame = cap.read()
                if ret:
                    current_frames[i] = frame

            if len(current_frames) == len(self.cameras):
                with self.frame_lock:
                    self.frames = current_frames.copy()

            time.sleep(0.01)

    def detect_and_match_features(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return None, None, kp1, kp2

        knn_matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        ratio_thresh = 0.75

        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            return None, None, kp1, kp2

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return src_pts, dst_pts, kp1, kp2

    def compute_homography_ransac(self, src_pts, dst_pts):
        if src_pts is None or dst_pts is None or len(src_pts) < 4:
            return None

        homography, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=2000,
            confidence=0.999,
        )

        if homography is not None:
            inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0
            if inlier_ratio < 0.3:
                return None

        return homography

    def select_best_calibration_frames(self, calibration_frames):
        best_score = 0
        best_frames = None

        for frames in calibration_frames:
            total_features = 0
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kp, _ = self.sift.detectAndCompute(gray, None)
                total_features += len(kp) if kp else 0

            if total_features > best_score:
                best_score = total_features
                best_frames = frames

        return best_frames

    def calibrate_cameras(self):
        print('Đang calibrate cameras... Hãy đảm bảo có vùng overlap giữa các camera')
        time.sleep(2)

        calibration_frames = []
        for _ in range(30):
            with self.frame_lock:
                current_frames = self.frames.copy()

            if len(current_frames) == len(self.cameras):
                sorted_frames = [current_frames[i] for i in sorted(current_frames.keys())]
                calibration_frames.append(sorted_frames)

            time.sleep(0.1)

        if not calibration_frames:
            raise ValueError('Không thể lấy frames để calibration')

        best_frames = self.select_best_calibration_frames(calibration_frames)
        if best_frames is None:
            raise ValueError('Không chọn được bộ frames tốt để calibration')

        print('Đang tính toán homography matrices...')
        reference_frame = best_frames[0]
        self.homographies = [np.eye(3, dtype=np.float32)]

        for i in range(1, len(best_frames)):
            src_pts, dst_pts, _, _ = self.detect_and_match_features(best_frames[i], reference_frame)
            if src_pts is None or dst_pts is None:
                raise ValueError(f'Không đủ features match cho camera {i}')

            homography = self.compute_homography_ransac(src_pts, dst_pts)
            if homography is None:
                raise ValueError(f'Không thể tính homography cho camera {i} (RANSAC failed)')

            self.homographies.append(homography)
            print(f'✓ Đã tính homography cho camera {i}')

        self.calculate_output_canvas(best_frames)
        self.is_calibrated = True
        print(f'✓ Calibration hoàn tất! Output size: {self.output_size}')

    def calculate_output_canvas(self, frames):
        h, w = frames[0].shape[:2]
        all_corners = []

        for i, homography in enumerate(self.homographies):
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped_corners = corners if i == 0 else cv2.perspectiveTransform(corners, homography)
            all_corners.extend(warped_corners.reshape(-1, 2))

        all_corners = np.array(all_corners)
        x_coords = all_corners[:, 0]
        y_coords = all_corners[:, 1]

        x_min, x_max = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
        y_min, y_max = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))

        self.translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        self.output_size = (x_max - x_min, y_max - y_min)

        for i in range(len(self.homographies)):
            self.homographies[i] = self.translation @ self.homographies[i]

    def stitch_frames_fast(self, frames):
        if not self.is_calibrated:
            return None

        result = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)
        mask = np.zeros((self.output_size[1], self.output_size[0]), dtype=np.uint8)

        for frame, homography in zip(frames, self.homographies):
            warped = cv2.warpPerspective(frame, homography, self.output_size)
            warped_mask = cv2.warpPerspective(
                np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255,
                homography,
                self.output_size,
            )

            valid_pixels = warped_mask > 0
            overlap_pixels = (mask > 0) & valid_pixels
            new_pixels = valid_pixels & (mask == 0)

            result[new_pixels] = warped[new_pixels]
            mask[new_pixels] = 255

            if np.any(overlap_pixels):
                result[overlap_pixels] = (
                    0.5 * result[overlap_pixels] + 0.5 * warped[overlap_pixels]
                ).astype(np.uint8)

        return result

    def detect_people(self, frame):
        if self.detector is None:
            return frame, 0, []

        results = self.detector.predict(frame, verbose=False, classes=[0], conf=self.person_conf)
        boxes = []
        annotated = frame.copy()

        if results and len(results) > 0:
            res = results[0]
            if res.boxes is not None:
                for box in res.boxes:
                    cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                    conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                    if cls_id != 0:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    boxes.append((x1, y1, x2, y2, conf))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f'person {conf:.2f}',
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        return annotated, len(boxes), boxes

    def overlay_status(self, frame, person_count, fps):
        cv2.rectangle(frame, (10, 10), (330, 95), (0, 0, 0), -1)
        cv2.putText(frame, f'FPS: {fps:.1f}', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f'Nguoi: {person_count}', (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, 'q: thoat | s: luu | r: recalibrate', (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame

    def run(self):
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        capture_thread.start()

        print('Đang khởi động cameras...')
        time.sleep(1)

        try:
            self.calibrate_cameras()
        except Exception as e:
            print(f'Lỗi calibration: {e}')
            print('Hãy đảm bảo:')
            print('- Các camera có vùng overlap')
            print('- Có đủ ánh sáng và texture trong scene')
            self.running = False
            return

        print("Bắt đầu real-time stitching + YOLO. Nhấn 'q' để thoát, 's' để lưu ảnh, 'r' để recalibrate")

        fps_counter = 0
        start_time = time.time()
        fps = 0.0

        while True:
            with self.frame_lock:
                current_frames = self.frames.copy()

            if len(current_frames) == len(self.cameras):
                sorted_frames = [current_frames[i] for i in sorted(current_frames.keys())]
                stitched = self.stitch_frames_fast(sorted_frames)

                if stitched is not None:
                    self.frame_count += 1
                    if self.frame_count % self.detect_every == 0:
                        annotated, person_count, boxes = self.detect_people(stitched)
                        self.last_person_boxes = boxes
                        self.last_person_count = person_count
                        display_source = annotated
                    else:
                        display_source = stitched.copy()
                        for x1, y1, x2, y2, conf in self.last_person_boxes:
                            cv2.rectangle(display_source, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_source, f'person {conf:.2f}', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    fps_counter += 1
                    if fps_counter % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = fps_counter / elapsed if elapsed > 0 else 0.0
                        print(f'FPS: {fps:.1f} | People: {self.last_person_count}')

                    display_img = self.overlay_status(display_source, self.last_person_count, fps)

                    display_height = 600
                    aspect_ratio = display_img.shape[1] / display_img.shape[0]
                    display_width = int(display_height * aspect_ratio)
                    if display_width > 1200:
                        display_width = 1200
                        display_height = int(display_width / aspect_ratio)

                    display_img = cv2.resize(display_img, (display_width, display_height))
                    cv2.imshow('Real-time Camera Stitching + YOLO People Count', display_img)

                    for i, frame in enumerate(sorted_frames):
                        small_frame = cv2.resize(frame, (320, 240))
                        cv2.imshow(f'Camera {i}', small_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = int(time.time())
                        filename = f'stitched_yolo_{timestamp}.jpg'
                        cv2.imwrite(filename, display_source)
                        print(f'Đã lưu ảnh: {filename}')
                    elif key == ord('r'):
                        print('Đang recalibrate...')
                        try:
                            self.calibrate_cameras()
                            print('Recalibration hoàn tất!')
                        except Exception as e:
                            print(f'Lỗi recalibration: {e}')

        self.cleanup()

    def cleanup(self):
        self.running = False
        for cap in self.cameras:
            cap.release()
        cv2.destroyAllWindows()
        print('Đã dọn dẹp resources')


def main():
    parser = argparse.ArgumentParser(description='Real-time Camera Stitching + YOLO People Counting')
    parser.add_argument('--cameras', nargs='+', type=int, default=[0, 1], help='Camera indices to use (default: 0 1)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--person-conf', type=float, default=0.35, help='Confidence threshold for person detection')
    parser.add_argument('--detect-every', type=int, default=3, help='Run YOLO every N stitched frames')

    args = parser.parse_args()

    try:
        stitcher = CameraStitcherYOLO(
            camera_indices=args.cameras,
            debug=args.debug,
            yolo_model=args.yolo_model,
            person_conf=args.person_conf,
            detect_every=args.detect_every,
        )
        stitcher.run()
    except Exception as e:
        print(f'Lỗi: {e}')
        print('Hướng dẫn sử dụng:')
        print('1. Đảm bảo có ít nhất 2 camera được kết nối')
        print('2. Cài dependencies: pip install ultralytics opencv-python numpy')
        print('3. Chạy: python main.py --cameras 0 1')
        print('4. Đảm bảo các camera có vùng overlap để calibration')
        print('5. Hệ thống sẽ stitch trước, sau đó YOLO đếm số người trên ảnh ghép')


if __name__ == '__main__':
    main()
