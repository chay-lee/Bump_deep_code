import numpy as np

config = {
    # --- Execution & Hardware ---
    "is_train":     True,           # 실행 모드 (True: 학습, False: 테스트)
    "device":       "cuda",         # 연산 장치 ("cuda" or "cpu")
    "cuda_mode":    "single_gpu",   # GPU 사용 모드 ("single_gpu" or "multi_gpu")
    "seed":         42,             # 재현성 시드 (None: 랜덤)
    "verbose":      False,          # 상세 정보 출력 여부

    # --- Project & Model Paths ---
    "model_feature": "m1_double",    # 모델 특징/버전명
    "model_root":    "/home/dspl/CY/Model", # 모델 저장 최상위 경로
    "model_mode":    "_final",               # 불러올 모델 가중치 이름 (e.g., '_final')
    "resume":        False,                  # 학습 재개 여부
    "resume_path":   "/home/dspl/CY/Model/0c_2_0.2_0.1_0/0c_2_0.2_0.1_0_final.pt",

    # --- Data & Domain (3D) ---
    "dataset_path": "/home/dspl/CY/Bump/Dataset/ATI_coin_bump_2", # 데이터셋 경로
    "scan_params":  [56, 2, 2 * np.sqrt(2)],                  # 3D 스캔 파라미터 [D, H, W]
    "y_range":      [0, 32],                                  # y축 범위
    "depth":        56,                                       # 3D 볼륨의 깊이(D)

    # --- Training Hyperparameters ---
    "epochs":       150,     # 총 학습 에폭
    "initial_lr":   8e-3,    # 초기 학습률
    "batch_size":   64,      # 학습 배치 사이즈
    "eval_epoch":   5,       # 검증(Validation) 주기
    "val_patience": 3,       # Early Stopping 기준 (Patience)

    # --- Evaluation & Testing ---
    "data_type":       "repeat_test", # 데이터 타입
    "repeat_test":     False,         # 반복 테스트 수행 여부
    "batch_size_test": 32,            # 테스트 배치 사이즈
    "lambda_val_1" : 0.0,
    "lambda_val_2" : 0.0,
    "rand_5" : False,
    "coordconv" : True
}