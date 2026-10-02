import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk


def color_quantize(img, K=8):
    """
    K-평균(k-means)을 이용한 색상 양자화.
    img : BGR 형태의 NumPy 배열
    K   : 색상 팔레트의 개수
    """
    data = np.float32(img).reshape((-1, 3))  # (픽셀 수, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)  # 군집(클러스터) 중심값 (BGR)
    result = centers[labels.flatten()]
    result = result.reshape(img.shape)
    return result


def cartoonize_image(
    img_path,
    num_bilateral=5,
    median_ksize=7,
    use_morph=False,
    use_color_quant=False,
    K=8,
):
    """
    OpenCV를 사용해 이미지를 간단히 만화(카툰) 스타일로 변환하는 함수.
    모폴로지 연산(노이즈 제거) 및 색상 양자화(팔레트 K개) 옵션을 추가로 적용 가능.

    Parameters
    ----------
    img_path : str
        변환할 이미지의 경로
    num_bilateral : int
        Bilateral Filter 반복 적용 횟수
    median_ksize : int
        Median Blur 커널 크기
    use_morph : bool
        True로 설정 시, 모폴로지 연산(오픈 & 클로즈)으로 노이즈 제거
    use_color_quant : bool
        True로 설정 시, 색상 양자화로 색 영역을 더 단순화 (K-평균)
    K : int
        색상 양자화 시 사용할 팔레트 개수
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("이미지를 찾을 수 없습니다. 경로를 확인하세요.")

    # 1) Bilateral Filter를 여러 번 적용하여 색 영역 부드럽게
    cartoon_color = img.copy()
    for _ in range(num_bilateral):
        cartoon_color = cv2.bilateralFilter(
            cartoon_color, d=9, sigmaColor=75, sigmaSpace=75
        )

    # (옵션) 색상 양자화
    if use_color_quant:
        cartoon_color = color_quantize(cartoon_color, K=K)

    # 2) Grayscale 변환 후 Median Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, median_ksize)

    # 3) Adaptive Threshold를 이용한 윤곽(엣지) 검출
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
    )

    # (옵션) 모폴로지 연산으로 잔 노이즈 제거
    if use_morph:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_OPEN, kernel
        )  # 작은 흰 점(노이즈) 제거
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # 윤곽선의 구멍 메우기

    # 4) 색 영역과 엣지를 합성
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(cartoon_color, edges_colored)

    return cartoon


class CartoonApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Cartoon Image App")

        # 원본/결과 이미지를 저장할 OpenCV BGR 배열
        self.original_img_cv = None
        self.cartoon_result_cv = None

        # ========================= 상단 제어 영역 =========================
        self.btn_frame = tk.Frame(self.master)
        self.btn_frame.pack(pady=5)

        # "이미지 선택" 버튼
        self.select_btn = tk.Button(
            self.btn_frame, text="이미지 선택", command=self.select_file
        )
        self.select_btn.pack(side="left", padx=5)

        # "결과 저장" 버튼
        self.save_btn = tk.Button(
            self.btn_frame, text="결과 저장", command=self.save_file
        )
        self.save_btn.pack(side="left", padx=5)

        # "다시하기" 버튼
        self.reset_btn = tk.Button(
            self.btn_frame, text="다시하기", command=self.reset_app
        )
        self.reset_btn.pack(side="left", padx=5)

        # ==================== 옵션(체크박스 & 슬라이더) 영역 ====================
        self.option_frame = tk.Frame(self.master)
        self.option_frame.pack(pady=5)

        # 모폴로지(노이즈 제거) 사용 여부
        self.use_morph_var = tk.BooleanVar(value=False)
        self.morph_check = tk.Checkbutton(
            self.option_frame, text="노이즈 제거(모폴로지)", variable=self.use_morph_var
        )
        self.morph_check.pack(side="left", padx=5)

        # 색상 양자화 사용 여부
        self.use_quant_var = tk.BooleanVar(value=False)
        self.quant_check = tk.Checkbutton(
            self.option_frame, text="색상 양자화", variable=self.use_quant_var
        )
        self.quant_check.pack(side="left", padx=5)

        # K값 슬라이더 (2 ~ 16)
        self.K_scale = tk.Scale(
            self.option_frame,
            from_=2,
            to=16,
            orient="horizontal",
            label="K(팔레트 개수)",
            length=200,
        )
        self.K_scale.set(8)  # 기본값 8
        self.K_scale.pack(side="left", padx=5)

        # ========================= 이미지 표시 영역 =========================
        self.image_frame = tk.Frame(self.master)
        self.image_frame.pack()

        # 원본 이미지를 표시할 라벨
        self.original_label = tk.Label(self.image_frame, text="원본 이미지 미리보기")
        self.original_label.pack(side="left", padx=10)

        # 카툰 이미지를 표시할 라벨
        self.cartoon_label = tk.Label(self.image_frame, text="카툰 이미지 미리보기")
        self.cartoon_label.pack(side="right", padx=10)

    def select_file(self):
        """파일 대화상자를 통해 이미지 파일을 선택하고 카툰 변환 이미지를 보여줌"""
        file_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.gif"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            # 원본 이미지 로드 (OpenCV BGR)
            self.original_img_cv = cv2.imread(file_path)

            # 체크박스 & 슬라이더 옵션 읽기
            use_morph = self.use_morph_var.get()
            use_color_quant = self.use_quant_var.get()
            K_value = self.K_scale.get()

            # 카툰 변환
            self.cartoon_result_cv = cartoonize_image(
                file_path,
                use_morph=use_morph,
                use_color_quant=use_color_quant,
                K=K_value,
            )

            # ---- Tk Label에 표시하기 위해 PIL Image로 변환 (BGR -> RGB) ----
            if self.original_img_cv is not None:
                original_img_rgb = cv2.cvtColor(self.original_img_cv, cv2.COLOR_BGR2RGB)
                original_pil = Image.fromarray(original_img_rgb)
                self.original_tk = ImageTk.PhotoImage(original_pil)
                self.original_label.config(image=self.original_tk, text="")
                self.original_label.image = self.original_tk  # 참조 유지

            if self.cartoon_result_cv is not None:
                cartoon_img_rgb = cv2.cvtColor(
                    self.cartoon_result_cv, cv2.COLOR_BGR2RGB
                )
                cartoon_pil = Image.fromarray(cartoon_img_rgb)
                self.cartoon_tk = ImageTk.PhotoImage(cartoon_pil)
                self.cartoon_label.config(image=self.cartoon_tk, text="")
                self.cartoon_label.image = self.cartoon_tk  # 참조 유지

        else:
            print("파일 선택이 취소되었습니다.")

    def save_file(self):
        """결과 이미지를 원하는 경로에 저장"""
        if self.cartoon_result_cv is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All Files", "*.*")],
            )
            if file_path:
                cv2.imwrite(file_path, self.cartoon_result_cv)
                print(f"결과 이미지가 저장되었습니다: {file_path}")
        else:
            print("저장할 카툰 이미지가 없습니다. 먼저 이미지를 선택하세요.")

    def reset_app(self):
        """다시하기 버튼 클릭 시, 초기 상태로 복원"""
        # OpenCV 이미지 변수 초기화
        self.original_img_cv = None
        self.cartoon_result_cv = None

        # Label을 다시 텍스트만 보여주는 상태로 바꿔줍니다.
        self.original_label.config(image="", text="원본 이미지 미리보기")
        self.original_label.image = None

        self.cartoon_label.config(image="", text="카툰 이미지 미리보기")
        self.cartoon_label.image = None

        # 체크박스/슬라이더 기본 상태로 초기화
        self.use_morph_var.set(False)
        self.use_quant_var.set(False)
        self.K_scale.set(8)

        print("초기 상태로 돌아갔습니다.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CartoonApp(root)
    root.mainloop()
