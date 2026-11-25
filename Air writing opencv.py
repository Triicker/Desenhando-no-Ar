"""Air Writing com MediaPipe - desenhe no ar usando a webcam.

Controles:
  - Encoste polegar + indicador para desenhar
  - Separe polegar + indicador para pausar
  - Levante indicador + mindinho para disparar uma teia
  - Abra a mão para carregar um orbe de energia
  - Pisque 2 vezes o olho direito para trocar a cor do orbe
  - 'c' limpa a tela
  - 'q' sai
"""

from pathlib import Path
import math
import random
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_MODEL_PATH = Path(__file__).with_name("face_landmarker.task")

PINCH_ON_RATIO = 0.18
PINCH_OFF_RATIO = 0.26
SMOOTHING = 0.35
MAX_JUMP_RATIO = 0.42

CLEAR_COOLDOWN_SECONDS = 1.2
FINGER_UP_MARGIN = 0.025
THUMB_OPEN_MARGIN = 0.035

WEB_COOLDOWN_SECONDS = 0.45
WEB_EFFECT_DURATION_SECONDS = 0.42
WEB_LENGTH_RATIO = 6.0
WEB_FLICK_SPEED_RATIO = 0.12
WEB_BRANCH_COUNT = 9

POWER_GROW_SPEED = 0.08
POWER_FADE_SPEED = 0.12
POWER_MIN_RADIUS_RATIO = 0.34
POWER_MAX_RADIUS_RATIO = 0.72
POWER_PARTICLE_COUNT = 18
POWER_COLORS = [
    ((255, 245, 210), (255, 170, 50), (255, 80, 0)),
    ((255, 255, 255), (255, 210, 70), (30, 220, 255)),
    ((255, 245, 255), (210, 80, 255), (100, 20, 255)),
    ((230, 255, 235), (80, 255, 160), (20, 160, 80)),
]

RIGHT_BLINK_SCORE_THRESHOLD = 0.55
DOUBLE_BLINK_WINDOW_SECONDS = 1.0
COLOR_CHANGE_COOLDOWN_SECONDS = 0.8


def ensure_file(path: Path, url: str, label: str) -> None:
    if path.exists():
        return

    print(f"Baixando modelo do MediaPipe {label}...")
    urllib.request.urlretrieve(url, path)
    print(f"Modelo salvo em: {path}")


def ensure_models() -> None:
    ensure_file(MODEL_PATH, MODEL_URL, "Hand Landmarker")
    ensure_file(FACE_MODEL_PATH, FACE_MODEL_URL, "Face Landmarker")


def distance(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def landmark_to_point(landmark, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(landmark.x, 0.0, 1.0) * width)
    y = int(np.clip(landmark.y, 0.0, 1.0) * height)
    return x, y


def smooth_point(
    previous: tuple[float, float] | None,
    current: tuple[int, int],
) -> tuple[float, float]:
    if previous is None:
        return float(current[0]), float(current[1])

    x = previous[0] + (current[0] - previous[0]) * SMOOTHING
    y = previous[1] + (current[1] - previous[1]) * SMOOTHING
    return x, y


def draw_hand_landmarks(frame, landmarks, width: int, height: int) -> None:
    points = [landmark_to_point(landmark, width, height) for landmark in landmarks]

    for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[connection.start],
            points[connection.end],
            (80, 190, 255),
            2,
        )

    for point in points:
        cv2.circle(frame, point, 3, (30, 30, 30), -1)


def finger_is_up(landmarks, tip_id: int, pip_id: int) -> bool:
    return landmarks[tip_id].y < landmarks[pip_id].y - FINGER_UP_MARGIN


def clear_gesture_detected(landmarks) -> bool:
    index_up = finger_is_up(landmarks, 8, 6)
    middle_up = finger_is_up(landmarks, 12, 10)
    ring_up = finger_is_up(landmarks, 16, 14)
    pinky_up = finger_is_up(landmarks, 20, 18)
    return middle_up and not index_up and not ring_up and not pinky_up


def hand_open_detected(landmarks) -> bool:
    index_up = finger_is_up(landmarks, 8, 6)
    middle_up = finger_is_up(landmarks, 12, 10)
    ring_up = finger_is_up(landmarks, 16, 14)
    pinky_up = finger_is_up(landmarks, 20, 18)
    thumb_open = abs(landmarks[4].x - landmarks[2].x) > THUMB_OPEN_MARGIN
    return index_up and middle_up and ring_up and pinky_up and thumb_open


def web_gesture_detected(landmarks) -> bool:
    index_up = finger_is_up(landmarks, 8, 6)
    middle_up = finger_is_up(landmarks, 12, 10)
    ring_up = finger_is_up(landmarks, 16, 14)
    pinky_up = finger_is_up(landmarks, 20, 18)
    return index_up and pinky_up and not middle_up and not ring_up


def midpoint(point_a: tuple[int, int], point_b: tuple[int, int]) -> tuple[int, int]:
    return (point_a[0] + point_b[0]) // 2, (point_a[1] + point_b[1]) // 2


def normalized_vector(
    point_a: tuple[int, int],
    point_b: tuple[int, int],
) -> tuple[float, float]:
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    length = math.hypot(dx, dy)
    if length < 1.0:
        return 0.0, -1.0
    return dx / length, dy / length


def make_web_effect(
    origin: tuple[int, int],
    direction: tuple[float, float],
    hand_scale: float,
    width: int,
    height: int,
    created_at: float,
) -> dict:
    length = max(hand_scale * WEB_LENGTH_RATIO, max(width, height) * 0.75)
    target = (
        int(origin[0] + direction[0] * length),
        int(origin[1] + direction[1] * length),
    )
    perpendicular = -direction[1], direction[0]

    branches = []
    for index in range(WEB_BRANCH_COUNT):
        progress = 0.18 + index * 0.075
        side = -1 if index % 2 == 0 else 1
        spread = random.uniform(0.10, 0.24) * length * side
        forward = random.uniform(-0.03, 0.06) * length
        branches.append((progress, spread, forward))

    return {
        "origin": origin,
        "target": target,
        "direction": direction,
        "perpendicular": perpendicular,
        "branches": branches,
        "created_at": created_at,
        "duration": WEB_EFFECT_DURATION_SECONDS,
    }


def interpolate(
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    amount: float,
) -> tuple[int, int]:
    return (
        int(point_a[0] + (point_b[0] - point_a[0]) * amount),
        int(point_a[1] + (point_b[1] - point_a[1]) * amount),
    )


def average_point(points: list[tuple[int, int]]) -> tuple[int, int]:
    x = sum(point[0] for point in points) / len(points)
    y = sum(point[1] for point in points) / len(points)
    return int(x), int(y)


def palm_center(landmarks, width: int, height: int) -> tuple[int, int]:
    palm_ids = [0, 5, 9, 13, 17]
    return average_point(
        [landmark_to_point(landmarks[landmark_id], width, height) for landmark_id in palm_ids]
    )


def render_web_effects(frame, web_effects: list[dict], now: float) -> list[dict]:
    active_effects = []

    for effect in web_effects:
        elapsed = now - effect["created_at"]
        duration = effect["duration"]
        if elapsed > duration:
            continue

        active_effects.append(effect)
        progress = min(elapsed / duration, 1.0)
        fade = 1.0 - progress
        origin = effect["origin"]
        target = effect["target"]
        visible_end = interpolate(origin, target, progress)
        direction = effect["direction"]
        perpendicular = effect["perpendicular"]
        color = (
            int(200 + 40 * fade),
            int(225 + 25 * fade),
            255,
        )
        thickness = max(1, int(4 * fade) + 1)

        cv2.line(frame, origin, visible_end, color, thickness, cv2.LINE_AA)
        cv2.circle(frame, origin, max(3, int(9 * fade)), color, 1, cv2.LINE_AA)

        for branch_progress, spread, forward in effect["branches"]:
            if branch_progress > progress:
                continue

            branch_start = interpolate(origin, target, branch_progress)
            branch_end = (
                int(
                    branch_start[0]
                    + perpendicular[0] * spread * fade
                    + direction[0] * forward
                ),
                int(
                    branch_start[1]
                    + perpendicular[1] * spread * fade
                    + direction[1] * forward
                ),
            )
            cv2.line(frame, branch_start, branch_end, color, 1, cv2.LINE_AA)

    return active_effects


def blend_color(
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    return (
        int(color_a[0] + (color_b[0] - color_a[0]) * amount),
        int(color_a[1] + (color_b[1] - color_a[1]) * amount),
        int(color_a[2] + (color_b[2] - color_a[2]) * amount),
    )


def draw_alpha_circle(
    target,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: float,
    thickness: int = -1,
) -> None:
    overlay = np.zeros_like(target)
    cv2.circle(overlay, center, radius, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, target, 1.0, 0, dst=target)


def render_power_orb(
    frame,
    center: tuple[int, int],
    hand_scale: float,
    charge: float,
    color_index: int,
    now: float,
) -> None:
    if charge <= 0.01:
        return

    core_color, mid_color, edge_color = POWER_COLORS[color_index % len(POWER_COLORS)]
    radius = int(
        hand_scale
        * (
            POWER_MIN_RADIUS_RATIO
            + (POWER_MAX_RADIUS_RATIO - POWER_MIN_RADIUS_RATIO) * charge
        )
    )
    radius = max(radius, 12)
    pulse = 0.5 + 0.5 * math.sin(now * 12.0)
    glow_radius = int(radius * (1.55 + 0.18 * pulse))

    draw_alpha_circle(frame, center, glow_radius, edge_color, 0.12 * charge)
    draw_alpha_circle(frame, center, int(radius * 1.25), mid_color, 0.20 * charge)
    draw_alpha_circle(frame, center, radius, blend_color(mid_color, core_color, 0.30), 0.42)
    draw_alpha_circle(frame, center, max(5, int(radius * 0.42)), core_color, 0.62)

    for ring_index in range(3):
        angle = now * (130 + ring_index * 45) * (1 if ring_index % 2 == 0 else -1)
        axes = (
            max(4, int(radius * (1.05 + ring_index * 0.10))),
            max(3, int(radius * (0.38 + ring_index * 0.08))),
        )
        cv2.ellipse(
            frame,
            center,
            axes,
            angle,
            15,
            330,
            blend_color(core_color, edge_color, ring_index * 0.25),
            max(1, int(2 + charge)),
            cv2.LINE_AA,
        )

    for particle_index in range(POWER_PARTICLE_COUNT):
        orbit = now * (3.8 + (particle_index % 5) * 0.35) + particle_index * 2.399
        orbit_radius = radius * (0.45 + (particle_index % 7) * 0.12)
        px = int(center[0] + math.cos(orbit) * orbit_radius)
        py = int(center[1] + math.sin(orbit * 1.27) * orbit_radius * 0.58)
        particle_radius = 1 + int((particle_index % 3) * charge)
        cv2.circle(frame, (px, py), particle_radius, core_color, -1, cv2.LINE_AA)


def blendshape_score(face_result, category_name: str) -> float:
    if not face_result.face_blendshapes:
        return 0.0

    for category in face_result.face_blendshapes[0]:
        if category.category_name == category_name:
            return category.score
    return 0.0


def update_double_blink(
    right_eye_closed: bool,
    right_eye_was_closed: bool,
    blink_times: list[float],
    now: float,
) -> tuple[bool, list[float], bool]:
    double_blink_detected = False

    if right_eye_was_closed and not right_eye_closed:
        blink_times.append(now)
        blink_times = [
            blink_time
            for blink_time in blink_times
            if now - blink_time <= DOUBLE_BLINK_WINDOW_SECONDS
        ]
        if len(blink_times) >= 2:
            double_blink_detected = True
            blink_times = []

    return right_eye_closed, blink_times, double_blink_detected


def main() -> None:
    ensure_models()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERRO: Não foi possível acessar a webcam.")
        print("Feche Zoom/Teams/Meet e confira as permissões de câmera do Windows.")
        return

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.60,
        min_face_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    canvas = None
    prev_point = None
    filtered_point = None
    drawing_active = False
    last_clear_time = 0.0
    last_web_time = 0.0
    web_gesture_was_active = False
    previous_wrist = None
    web_effects = []
    power_charge = 0.0
    power_center = None
    power_hand_scale = 1.0
    power_color_index = 0
    right_eye_was_closed = False
    right_blink_times = []
    last_color_change_time = 0.0
    start_time = time.perf_counter()
    last_timestamp_ms = -1

    print("Câmera iniciada. Belisque indicador + polegar para desenhar.")
    print("Dedo médio levantado sozinho limpa a tela.")
    print("Indicador + mindinho levantados dispara uma teia.")
    print("Mão aberta carrega o orbe. Pisque 2 vezes o olho direito para trocar a cor.")
    print("Pressione 'c' para limpar ou 'q' para sair.")

    with (
        HandLandmarker.create_from_options(hand_options) as hand_landmarker,
        FaceLandmarker.create_from_options(face_options) as face_landmarker,
    ):
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("ERRO: A câmera abriu, mas falhou ao capturar o frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            if canvas is None:
                canvas = np.zeros_like(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            status = "PAUSADO"
            status_color = (0, 180, 255)
            web_shot = False
            color_changed = False
            now = time.perf_counter()

            right_blink_score = blendshape_score(face_result, "eyeBlinkRight")
            right_eye_closed = right_blink_score > RIGHT_BLINK_SCORE_THRESHOLD
            (
                right_eye_was_closed,
                right_blink_times,
                double_blink_detected,
            ) = update_double_blink(
                right_eye_closed,
                right_eye_was_closed,
                right_blink_times,
                now,
            )
            if (
                double_blink_detected
                and now - last_color_change_time > COLOR_CHANGE_COOLDOWN_SECONDS
            ):
                power_color_index = (power_color_index + 1) % len(POWER_COLORS)
                last_color_change_time = now
                status = "COR ALTERADA"
                status_color = POWER_COLORS[power_color_index][1]
                color_changed = True

            if hand_result.hand_landmarks:
                landmarks = hand_result.hand_landmarks[0]
                draw_hand_landmarks(frame, landmarks, width, height)

                index_tip = landmark_to_point(landmarks[8], width, height)
                thumb_tip = landmark_to_point(landmarks[4], width, height)
                pinky_tip = landmark_to_point(landmarks[20], width, height)
                wrist = landmark_to_point(landmarks[0], width, height)
                middle_mcp = landmark_to_point(landmarks[9], width, height)

                hand_scale = max(distance(wrist, middle_mcp), 1.0)
                pinch_ratio = distance(index_tip, thumb_tip) / hand_scale

                if drawing_active:
                    drawing_active = pinch_ratio < PINCH_OFF_RATIO
                else:
                    drawing_active = pinch_ratio < PINCH_ON_RATIO

                if (
                    clear_gesture_detected(landmarks)
                    and now - last_clear_time > CLEAR_COOLDOWN_SECONDS
                ):
                    canvas = np.zeros_like(frame)
                    prev_point = None
                    drawing_active = False
                    last_clear_time = now
                    status = "TELA LIMPA"
                    status_color = (0, 180, 255)

                web_gesture_active = web_gesture_detected(landmarks)
                wrist_speed_ratio = 0.0
                if previous_wrist is not None:
                    wrist_speed_ratio = distance(previous_wrist, wrist) / hand_scale

                web_gesture_started = web_gesture_active and not web_gesture_was_active
                web_flick_detected = (
                    web_gesture_active and wrist_speed_ratio > WEB_FLICK_SPEED_RATIO
                )
                if (
                    (web_gesture_started or web_flick_detected)
                    and now - last_web_time > WEB_COOLDOWN_SECONDS
                ):
                    web_origin = midpoint(index_tip, pinky_tip)
                    direction = normalized_vector(wrist, web_origin)
                    web_effects.append(
                        make_web_effect(
                            web_origin,
                            direction,
                            hand_scale,
                            width,
                            height,
                            now,
                        )
                    )
                    last_web_time = now
                    status = "TEIA!"
                    status_color = (255, 255, 255)
                    web_shot = True

                web_gesture_was_active = web_gesture_active
                previous_wrist = wrist

                if hand_open_detected(landmarks):
                    power_charge = min(1.0, power_charge + POWER_GROW_SPEED)
                    power_center = palm_center(landmarks, width, height)
                    power_hand_scale = hand_scale
                    if not web_shot and status == "PAUSADO":
                        status = "CARREGANDO"
                        status_color = POWER_COLORS[power_color_index][1]
                else:
                    power_charge = max(0.0, power_charge - POWER_FADE_SPEED)

                filtered_point = smooth_point(filtered_point, index_tip)
                pen_point = int(filtered_point[0]), int(filtered_point[1])

                if drawing_active:
                    if not web_shot and not color_changed:
                        status = "DESENHANDO"
                        status_color = (0, 255, 0)
                    if prev_point is not None:
                        jump = distance(prev_point, pen_point)
                        if jump < hand_scale * MAX_JUMP_RATIO:
                            cv2.line(canvas, prev_point, pen_point, (0, 255, 0), 5)
                    prev_point = pen_point
                else:
                    prev_point = None

                cv2.circle(frame, index_tip, 10, status_color, -1)
                cv2.circle(frame, thumb_tip, 8, (255, 120, 0), -1)
                cv2.line(frame, index_tip, thumb_tip, status_color, 2)
            else:
                prev_point = None
                filtered_point = None
                drawing_active = False
                previous_wrist = None
                web_gesture_was_active = False
                power_charge = max(0.0, power_charge - POWER_FADE_SPEED)

            output = cv2.addWeighted(frame, 1, canvas, 1, 0)
            web_effects = render_web_effects(output, web_effects, time.perf_counter())
            if power_center is not None:
                render_power_orb(
                    output,
                    power_center,
                    power_hand_scale,
                    power_charge,
                    power_color_index,
                    time.perf_counter(),
                )
            cv2.rectangle(output, (0, 0), (width, 72), (20, 20, 20), -1)
            cv2.putText(
                output,
                "Belisque=desenhar | indicador+mindinho=teia | mao aberta=orbe",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                f"{status}   olho direito x2=muda cor   medio=limpar   c=limpar   q=sair",
                (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Air Writing", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                canvas = np.zeros_like(frame)
                prev_point = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
