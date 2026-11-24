
from pathlib import Path

import cv2
import numpy as np

from Lab2.tracker import ObjectTracker, TrackingResult


def process_video(
    input_path: Path,
    output_path: Path,
    method: str = "sift",
    show_video: bool = True
) -> None:

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print(f"Не удалось открыть: {input_path}")
        return

    writer = None

    try:
        ret, first_frame = cap.read()

        if not ret:
            print(f"Видео не содержит кадров: {input_path}")
            return

        height, width = first_frame.shape[:2]

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        tracker = ObjectTracker(
            reference=first_frame,
            method=method
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError(f"Не удалось создать видео: {output_path}")

        corners = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])

        first_result = TrackingResult(
            found=True,
            corners=corners,
            inliers=len(tracker.ref_kp)
        )

        total_frames = 0
        detected_frames = 0

        frame = first_frame

        while True:
            if total_frames == 0:
                result = first_result
            else:
                result = tracker.track(frame)

            output_frame = tracker.draw(frame, result,"Object")
            writer.write(output_frame)

            total_frames += 1
            detected_frames += int(result.found)

            if show_video:
                cv2.imshow(f"Tracking: {input_path.name} | {method.upper()}", output_frame)
                key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF

                if key == ord("q"):
                    break

            ret, frame = cap.read()

            if not ret:
                break

        print(f"\nВидео: {input_path.name}")
        print(f"Алгоритм: {method.upper()}")
        print(f"Всего кадров: {total_frames}")
        print(f"Объект обнаружен: {detected_frames}")
        print(f"Не обнаружен: {total_frames - detected_frames}")
        print(f"Сохранено: {output_path}")

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        if show_video:
            cv2.destroyAllWindows()


def main() -> None:

    mode = "all"

    input_dir = Path("data")
    output_dir = Path("results")

    video_name = "mona-lisa-blur.avi"
    method = "sift"

    methods = ["sift", "orb"]
    show_video = True
    supported_formats = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "single":
        input_path = input_dir / video_name

        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        output_path = output_dir / f"{input_path.stem}_tracked_{method}.mp4"

        process_video(
            input_path=input_path,
            output_path=output_path,
            method=method,
            show_video=show_video
        )

    elif mode == "all":
        videos = sorted(
            path for path in input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in supported_formats
        )

        if not videos:
            print("В папке data нет видеофайлов")
            return

        print(f"Найдено видео: {len(videos)}")
        print(f"Алгоритмы: {', '.join(methods).upper()}")

        for video in videos:
            for current_method in methods:

                output_path = output_dir / (f"{video.stem}_tracked_{current_method}.mp4")

                print(
                    f"\nОбработка: {video.name} "
                    f"[{current_method.upper()}]"
                )

                try:
                    process_video(
                        input_path=video,
                        output_path=output_path,
                        method=current_method,
                        show_video=show_video
                    )
                except (ValueError, RuntimeError, cv2.error) as exc:
                    print(f"Ошибка обработки: {exc}")

    else:
        raise ValueError("mode должен быть 'all' или 'single'")


if __name__ == "__main__":
    main()
