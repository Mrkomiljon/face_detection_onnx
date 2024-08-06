import os
import cv2
import numpy as np
import onnxruntime
from typing import Tuple
import argparse

__all__ = ["SCRFD"]

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def draw_corners(image, bbox):
    x1, y1, x2, y2, _ = bbox.astype(np.int32)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

def draw_keypoints(image, kps):
    for i in range(kps.shape[0]):
        cv2.circle(image, tuple(kps[i].astype(np.int32)), 2, (0, 255, 0), -1)

class SCRFD:
    def __init__(self, model_path: str, input_size: Tuple[int] = (640, 640), conf_thres: float = 0.3, iou_thres: float = 0.4) -> None:
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

        self.mean = 127.5
        self.std = 128.0

        self.center_cache = {}

        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str):
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"] # "CUDAExecutionProvider", 
            )
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]
        except Exception as e:
            print(f"Failed to load the model: {e}")
            raise

    def forward(self, image, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(image.shape[0:2][::-1])

        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / self.std,
            input_size,
            (self.mean, self.mean, self.mean),
            swapRB=True
        )
        outputs = self.session.run(self.output_names, {self.input_names[0]: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = outputs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, image, max_num=0, metric="max"):
        width, height = self.input_size

        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = height / width
        if im_ratio > model_ratio:
            new_height = height
            new_width = int(new_height / im_ratio)
        else:
            new_width = width
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / image.shape[0]
        resized_image = cv2.resize(image, (new_width, new_height))

        det_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Handle 4-channel images by converting to 3-channel
        if resized_image.shape[2] == 4:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2BGR)
        
        det_image[:new_height, :new_width, :] = resized_image

        scores_list, bboxes_list, kpss_list = self.forward(det_image, self.conf_thres)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale

        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det, iou_thres=self.iou_thres)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            image_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - image_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - image_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = (area - offset_dist_squared * 2.0)  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets, iou_thres):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= iou_thres)[0]
            order = order[indices + 1]

        return keep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection using SCRFD")
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--video', type=str, help='Path to the video file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for capturing video')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Path to save the output video')

    args = parser.parse_args()

    detector = SCRFD(model_path="insightface/models/buffalo_l/det_10g.onnx")

    if args.image:
        # Process a single image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not open image {args.image}")
            exit()

        # Detect faces
        boxes_list, points_list = detector.detect(image)

        # Draw detected faces and keypoints
        for boxes in boxes_list:
            draw_corners(image, boxes)

        if points_list is not None:
            for points in points_list:
                draw_keypoints(image, points)

        # Save the result
        output_image_path = args.output if args.output.endswith(('.png', '.jpg', '.jpeg')) else args.output + ".jpg"
        cv2.imwrite(output_image_path, image)
        print(f"Output image saved to {output_image_path}")

        # Optionally, display the image
        cv2.imshow("FaceDetection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.video or args.camera is not None:
        # Video yoki kamera tanlash
        if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(args.camera)

        if not cap.isOpened():
            print("Error: Could not open video or camera.")
            exit()

        # VideoWriter uchun parametrlari
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # yoki 'XVID'
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:  # Kamera uchun FPS 0 bo'lishi mumkin
            fps = 30  # Oddiy kameralar uchun odatiy FPS
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = args.output if args.output.endswith('.mp4') else args.output + ".mp4"
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            boxes_list, points_list = detector.detect(frame)

            # Draw detected faces and keypoints
            for boxes in boxes_list:
                draw_corners(frame, boxes)

            if points_list is not None:
                for points in points_list:
                    draw_keypoints(frame, points)
            # Write the frame to the output video file
            out.write(frame)

            # Optionally, display the frame
            cv2.imshow("FaceDetection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()  # Yozuvni tugatish
        cv2.destroyAllWindows()

    else:
        print("Error: Please provide either an image path, a video file, or a camera index.")