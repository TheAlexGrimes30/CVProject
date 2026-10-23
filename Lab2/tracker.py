from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class TrackingResult:
    found: bool
    corners: np.ndarray | None = None
    matches: int = 0
    inliers: int = 0

class ObjectTracker:
    def __init__(
            self,
            reference: np.ndarray,
            method: str = "sift",
            ratio_threshold: float | None = None,
            min_matches: int = 10,
            min_inliers: int = 8,
            min_inlier_ratio: float = 0.40,
            ransac_threshold: float = 4.0
    ):
        if reference is None or reference.size == 0:
            raise ValueError("Первый кадр / эталонное изображение пустое")

        self.method = method.lower()

        if self.method == "sift":
            self.detector = cv2.SIFT.create(nfeatures=2000)
            norm = cv2.NORM_L2
            default_ratio = 0.75

        else:
            self.detector = cv2.ORB.create(nfeatures=2500)
            norm = cv2.NORM_HAMMING
            default_ratio = 0.80

        self.matcher = cv2.BFMatcher(norm, crossCheck=False)
        self.ratio_threshold = default_ratio if ratio_threshold is None else ratio_threshold

        if not 0 < self.ratio_threshold < 1:
            raise ValueError("ratio_threshold должен находиться в интервале (0, 1)")

        self.min_matches = max(4, min_matches)
        self.min_inliers = max(4, min_inliers)
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_threshold = ransac_threshold

        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_desc = self.detector.detectAndCompute(gray, None)

        if self.ref_desc is None or len(self.ref_kp) < self.min_matches:
            raise ValueError(
                "В эталоне слишком мало ключевых точек. "
                "Выберите более текстурированный объект или другую область."
            )

        h, w = gray.shape
        self.ref_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)

    @staticmethod
    def _valid_corners(corners: np.ndarray, shape: tuple[int, ...]) -> bool:

        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            return False

        polygon = corners.astype(np.float32)
        if not cv2.isContourConvex(polygon):
            return False

        lengths = np.linalg.norm(polygon - np.roll(polygon, -1, axis=0), axis=1)
        if np.any(lengths < 8):
            return False

        h, w = shape[:2]
        area = abs(cv2.contourArea(polygon))

        if not 0.003 * w * h <= area <= 1.2 * w * h:
            return False

        x, y = polygon[:, 0], polygon[:, 1]
        if x.max() < 0 or x.min() >= w or y.max() < 0 or y.min() >= h:
            return False
        if x.min() < -0.2 * w or x.max() > 1.2 * w:
            return False
        if y.min() < -0.2 * h or y.max() > 1.2 * h:
            return False

        return True

    def track(self, frame: np.ndarray) -> TrackingResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_desc = self.detector.detectAndCompute(gray, None)

        if frame_desc is None or len(frame_kp) < 2:
            return TrackingResult(found=False)

        pairs = self.matcher.knnMatch(self.ref_desc, frame_desc, k=2)

        good = [
            a for pair in pairs if len(pair) == 2
            for a, b in [pair]
            if a.distance < self.ratio_threshold * b.distance
        ]

        count = len(good)
        if count < self.min_matches:
            return TrackingResult(found=False, matches=count)

        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_threshold)

        if matrix is None or mask is None:
            return TrackingResult(found=False, matches=count)

        inliers = int(mask.sum())
        if inliers < self.min_inliers or inliers / count < self.min_inlier_ratio:
            return TrackingResult(found=False, matches=count, inliers=inliers)

        corners = cv2.perspectiveTransform(
            self.ref_corners, matrix
        ).reshape(4, 2)

        if not self._valid_corners(corners, frame.shape):
            return TrackingResult(found=False, matches=count, inliers=inliers)

        return TrackingResult(
            found=True,
            corners=corners,
            matches=count,
            inliers=inliers
        )

    @staticmethod
    def draw(
            frame: np.ndarray,
            result: TrackingResult,
            label: str = "Object"
    ) -> np.ndarray:
        output = frame.copy()

        if result.found and result.corners is not None:
            corners = np.rint(result.corners).astype(np.int32)
            cv2.polylines(
                output, [corners], True, (0, 255, 0), 3, cv2.LINE_AA
            )

            x = int(np.clip(corners[:, 0].min(), 5, max(5, output.shape[1] - 1)))
            y = int(np.clip(corners[:, 1].min() - 10, 25, max(25, output.shape[0] - 1)))

            cv2.putText(
                output,
                f"{label} | inliers: {result.inliers}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        else:
            cv2.putText(
                output,
                "Object not found",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        return output
