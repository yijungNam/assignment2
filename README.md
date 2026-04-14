# Assignment #2 - DeepXplore on CIFAR-10 ResNet50

## 프로젝트 구조

```
.
├── test.py           # 메인 실행 스크립트 (DeepXplore 실행 + 결과 저장)
├── deepxplore.py     # DeepXplore 핵심 로직 (PyTorch 구현)
├── models.py         # ResNet50 CIFAR-10 모델 정의 및 학습 유틸리티
├── visualize.py      # 결과 시각화 유틸리티
├── requirements.txt  # Python 의존성 목록
├── results/          # 실행 후 생성되는 결과 이미지 디렉토리
│   ├── disagreement_001.png ~ disagreement_010.png
│   └── summary.png
├── model_a.pth       # 학습 후 저장되는 Model A 가중치
└── model_b.pth       # 학습 후 저장되는 Model B 가중치
```

---

## 환경 설정

### 1. Python 환경 준비 (Python 3.8 이상 권장)

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. GPU 사용 (선택, 강력 권장)

CUDA가 설치된 환경에서는 자동으로 GPU를 사용  

```bash
# CUDA 버전 확인 후 해당 버전의 torch 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 실행 방법

### 기본 실행 (전체 학습 + DeepXplore)

```bash
python test.py
```

- CIFAR-10 자동 다운로드 → 두 모델 30 epochs 학습 → DeepXplore 200 seeds 탐색 → 결과 저장
- GPU 기준 약 60~90분 소요

### 빠른 테스트 (5 epochs, 50 seeds)

```bash
python test.py --quick
```

- 기능 검증용 빠른 실행 (GPU 기준 약 30분)

### 이미 학습된 모델 사용

```bash
# model_a.pth, model_b.pth가 존재하는 경우
python test.py --skip-train
```

### 기타 옵션

```bash
python test.py --seeds 50          # 탐색 시드 수 지정
python test.py --steps 200          # 최적화 반복 횟수 증가 (더 강한 perturbation)
python test.py --lambda 0.7         # Disagreement loss 비중 증가
python test.py --step-size 0.02     # Gradient step 크기 조정
python test.py --output-dir out/    # 결과 저장 디렉토리 변경
python test.py --force-retrain      # 모델 재학습 강제
```

---

## DeepXplore 구현 설명

### 원본 DeepXplore와의 차이점 및 수정사항

원본 DeepXplore (https://github.com/peikexin9/deepxplore)는 Keras/TensorFlow 기반으로 작성되었으며, MNIST/ImageNet 예제를 제공합니다.  
본 구현에서는 다음과 같이 수정하였습니다:

| 항목 | 원본 | 본 구현 |
|------|------|---------|
| 프레임워크 | Keras / TensorFlow 1.x | PyTorch 2.x |
| 모델 | LeNet, VGG, etc. | ResNet50 (CIFAR-10 수정 버전) |
| 입력 크기 | 28×28 (MNIST), 224×224 (ImageNet) | 32×32 (CIFAR-10) |
| Neuron coverage | Keras layer activation | PyTorch forward hook |
| 최적화 | TF gradient tape | PyTorch autograd |

### ResNet50 수정사항 (CIFAR-10 대응)

ResNet50은 원래 ImageNet(224×224)용으로 설계되어, CIFAR-10(32×32)에 직접 적용 시 공간 해상도가 너무 빠르게 줄어드는 문제가 있습니다.

```python
# 수정 1: 첫 번째 Conv layer - kernel 7x7 → 3x3, stride 7→1
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

# 수정 2: MaxPool 제거 (작은 이미지에서 정보 손실 방지)
model.maxpool = nn.Identity()

# 수정 3: 출력 클래스 수 - 1000 → 10
model.fc = nn.Linear(model.fc.in_features, 10)
```

### 두 모델의 차별화 (Differential Testing 유효성)

두 모델이 완전히 동일하면 differential testing이 의미없으므로, 다른 하이퍼파라미터로 학습:

- **Model A**: lr=0.1, weight_decay=5e-4 (표준 SGD 설정)
- **Model B**: lr=0.05, weight_decay=1e-3 (더 강한 정규화)

두 모델 모두 동일한 아키텍처(ResNet50)지만, 다른 초기화 random seed와 학습 경로를 거쳐 서로 다른 decision boundary 형성, 동일한 random seed 설정을 사용하여 실행 시 유사한 결과를 재현할 수 있도록 구성

---

## 결과 해석

- `results/disagreement_XXX.png`: 각 케이스별 (시드 이미지 / 생성된 이미지 / perturbation 비교)
- `results/summary.png`: 전체 결과 요약 (상위 5개 이미지 + coverage 막대그래프 + 클래스 분포)
- DeepXplore는 200개의 시드 입력 중 173개의 disagreement를 발견했으며, 이는 86.5%의 높은 불일치 비율을 보였다.
Neuron coverage는 Model A에서 93.69%, Model B에서 95.46%로 측정되었다.
불일치는 주로 시각적으로 유사한 클래스 간에서 발생했으며, 예를 들어 cat/dog/frog와 같은 동물 클래스나 airplane/ship과 같은 운송수단 클래스에서 빈번하게 나타났다.
---

## 참고문헌

- Pei, K., Cao, Y., Yang, J., & Jana, S. (2017). DeepXplore: Automated Whitebox Testing of Deep Learning Systems. *SOSP 2017*.
- GitHub: https://github.com/peikexin9/deepxplore
