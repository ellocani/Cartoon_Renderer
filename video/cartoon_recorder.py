import cv2
import numpy as np
import os
from datetime import datetime
import random
import math
import argparse


def cartoon_filter(frame):
    # 1. 이미지 크기 축소
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (width // 2, height // 2))

    # 2. 색상 감소
    small = small // 32 * 32 + 16

    # 3. 양방향 필터로 스무딩 처리
    smooth = cv2.bilateralFilter(small, d=5, sigmaColor=75, sigmaSpace=75)

    # 4. 원본 크기로 복원
    smooth = cv2.resize(smooth, (width, height))

    # 5. 엣지 검출
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
    )

    # 6. 최종 합성
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(smooth, edges)

    return cartoon


class Particle:
    def __init__(self, width, height, particle_type):
        self.width = width
        self.height = height
        self.reset(particle_type)

    def reset(self, particle_type):
        self.x = random.randint(0, self.width)
        self.y = random.randint(-50, 0)

        if particle_type == "snow":
            self.size = random.randint(2, 4)
            self.speed = random.uniform(1, 3)
            self.wind = random.uniform(-0.5, 0.5)
            self.color = (255, 255, 255)
            self.oscillation = random.uniform(0, 2 * math.pi)
            self.osc_speed = random.uniform(0.02, 0.05)

        elif particle_type == "rain":
            self.size = random.randint(1, 2)
            self.speed = random.uniform(10, 15)
            self.wind = random.uniform(2, 4)
            self.color = (200, 200, 255)
            self.length = random.randint(10, 15)

        else:  # flower
            self.size = random.randint(3, 5)
            self.speed = random.uniform(1, 2)
            self.wind = random.uniform(-1, 1)
            self.color = (
                random.randint(200, 255),
                random.randint(150, 200),
                random.randint(200, 255),
            )
            self.rotation = random.uniform(0, 2 * math.pi)
            self.rot_speed = random.uniform(-0.1, 0.1)

    def update(self, particle_type):
        if particle_type == "snow":
            self.y += self.speed
            self.x += self.wind
            self.x += math.sin(self.oscillation) * 0.5
            self.oscillation += self.osc_speed

        elif particle_type == "rain":
            self.y += self.speed
            self.x += self.wind

        else:  # flower
            self.y += self.speed
            self.x += self.wind
            self.rotation += self.rot_speed

        # 화면 벗어나면 재설정
        if self.y > self.height or self.x < 0 or self.x > self.width:
            self.reset(particle_type)

    def draw(self, frame, particle_type):
        if particle_type == "snow":
            cv2.circle(frame, (int(self.x), int(self.y)), self.size, self.color, -1)

        elif particle_type == "rain":
            start_point = (int(self.x), int(self.y))
            end_point = (int(self.x + self.wind), int(self.y + self.length))
            cv2.line(frame, start_point, end_point, self.color, 1)

        else:  # flower
            points = np.array(
                [
                    [
                        self.size * math.cos(self.rotation),
                        self.size * math.sin(self.rotation),
                    ],
                    [
                        self.size * math.cos(self.rotation + 2.513),
                        self.size * math.sin(self.rotation + 2.513),
                    ],
                    [
                        self.size * math.cos(self.rotation + 5.027),
                        self.size * math.sin(self.rotation + 5.027),
                    ],
                    [
                        self.size * math.cos(self.rotation + 7.54),
                        self.size * math.sin(self.rotation + 7.54),
                    ],
                    [
                        self.size * math.cos(self.rotation + 10.053),
                        self.size * math.sin(self.rotation + 10.053),
                    ],
                ],
                np.int32,
            )
            points = points + np.array([int(self.x), int(self.y)])
            cv2.fillPoly(frame, [points], self.color)


class ParticleSystem:
    def __init__(self, width, height):
        self.particles = []
        self.width = width
        self.height = height
        self.active_type = None

    def initialize_particles(self, particle_type, count=100):
        self.particles = [
            Particle(self.width, self.height, particle_type) for _ in range(count)
        ]
        self.active_type = particle_type

    def update_and_draw(self, frame):
        if not self.active_type:
            return

        for particle in self.particles:
            particle.update(self.active_type)
            particle.draw(frame, self.active_type)


def main():
    parser = argparse.ArgumentParser(description="비디오 카툰화 프로그램")
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="카메라 소스 (0: 웹캠, rtsp://...: RTSP 주소)",
    )
    args = parser.parse_args()

    video_dir = "Video"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # 카메라 소스 설정
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)  # 웹캠 번호

    # 카메라 연결
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 파티클 시스템 초기화
    particle_system = ParticleSystem(width, height)

    print(f"비디오 설정 - 너비: {width}, 높이: {height}, FPS: {fps}")

    if width == 0 or height == 0:
        print("카메라로부터 올바른 해상도를 가져올 수 없습니다.")
        width = 640
        height = 480
        print(f"기본 해상도로 설정합니다: {width}x{height}")

    if fps == 0 or fps > 60:
        fps = 30
        print(f"FPS를 기본값 {fps}로 설정합니다.")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = None
    frame_count = 0
    last_frame_time = datetime.now()

    print("Press [SPACE] to toggle Recording Mode.")
    print("Press [F] to toggle Flip Mode.")
    print("Press [C] to toggle Cartoon Filter.")
    print("Press [S] to toggle Snow Effect.")
    print("Press [R] to toggle Rain Effect.")
    print("Press [F] to toggle Flower Effect.")
    print("Press [ESC] to Exit.")

    is_recording = False
    is_flipped = False
    use_cartoon = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽어올 수 없습니다.")
            break

        save_frame = frame.copy()

        if use_cartoon:
            frame = cartoon_filter(frame)
            save_frame = cartoon_filter(save_frame)

        # 파티클 시스템 업데이트 및 그리기
        particle_system.update_and_draw(frame)
        if is_recording:
            particle_system.update_and_draw(save_frame)

        if is_flipped:
            frame = cv2.flip(frame, 1)
            save_frame = cv2.flip(save_frame, 1)

        if is_recording:
            cv2.circle(frame, (50, 50), 15, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "Record Mode",
                (70, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            if out is None:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = os.path.join(video_dir, f"video_{timestamp}.avi")
                    out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

                    if not out.isOpened():
                        print(f"비디오 파일을 생성할 수 없습니다: {video_path}")
                        print("코덱 설정을 확인해주세요.")
                        is_recording = False
                        continue
                    print(f"녹화 시작: {video_path}")
                    print(f"설정 - 코덱: XVID, FPS: 30, 해상도: {width}x{height}")
                    frame_count = 0
                except Exception as e:
                    print(f"비디오 저장 중 오류 발생: {str(e)}")
                    is_recording = False
                    continue

            try:
                if save_frame is not None and save_frame.size > 0:
                    current_time = datetime.now()
                    time_diff = (current_time - last_frame_time).total_seconds()

                    if time_diff >= 0.033:
                        out.write(save_frame)
                        frame_count += 1
                        last_frame_time = current_time
                else:
                    print("유효하지 않은 프레임입니다.")
                    is_recording = False
                    if out is not None:
                        out.release()
                        out = None
            except Exception as e:
                print(f"프레임 저장 중 오류 발생: {str(e)}")
                is_recording = False
                if out is not None:
                    out.release()
                    out = None
        else:
            cv2.putText(
                frame,
                "Preview Mode",
                (70, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            if out is not None:
                out.release()
                out = None

        cv2.imshow("VisionCam (Press ESC to exit)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # Space
            is_recording = not is_recording
        elif key == ord("f") or key == ord("F"):
            if particle_system.active_type == "flower":
                particle_system.active_type = None
            else:
                particle_system.initialize_particles("flower", 50)
        elif key == ord("s") or key == ord("S"):
            if particle_system.active_type == "snow":
                particle_system.active_type = None
            else:
                particle_system.initialize_particles("snow", 100)
        elif key == ord("r") or key == ord("R"):
            if particle_system.active_type == "rain":
                particle_system.active_type = None
            else:
                particle_system.initialize_particles("rain", 200)
        elif key == ord("c") or key == ord("C"):
            use_cartoon = not use_cartoon

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
