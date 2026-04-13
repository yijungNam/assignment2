"""
test.py
DeepXplore를 ResNet50 CIFAR-10 모델에 실행하고 결과를 저장하는 메인 스크립트.

실행 방법:
    python test.py                  # 전체 실행 (모델 학습 포함, ~30 epochs)
    python test.py --quick          # 빠른 테스트 (5 epochs, 50 seeds)
    python test.py --skip-train     # 저장된 모델 사용 (학습 생략)
    python test.py --seeds 100      # 탐색할 시드 수 지정

출력:
    - results/disagreement_001.png ~ results/disagreement_010.png : 개별 케이스
    - results/summary.png : 전체 결과 요약
    - model_a.pth, model_b.pth : 학습된 모델 가중치
"""

import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision

from models import load_or_train_models, CIFAR10_CLASSES, DEVICE
from deepxplore import DeepXplore
from visualize import save_disagreement_plots, save_summary_plot


def get_test_loader(batch_size=1):
    """배치 크기 1의 테스트 데이터로더 반환 (DeepXplore 시드용)"""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    return torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )


def print_banner():
    print("=" * 60)
    print("  DeepXplore - CIFAR-10 Differential Testing")
    print("  Assignment #2 | Reliable & Trustworthy AI")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run DeepXplore on CIFAR-10 ResNet50 models')
    parser.add_argument('--quick',       action='store_true',
                        help='빠른 테스트 모드 (5 epochs 학습, 50 seeds)')
    parser.add_argument('--skip-train',  action='store_true',
                        help='저장된 모델 가중치 사용 (학습 건너뜀)')
    parser.add_argument('--seeds',       type=int, default=None,
                        help='탐색할 시드 수 (기본: quick=50, full=200)')
    parser.add_argument('--steps',       type=int, default=100,
                        help='입력 생성 반복 횟수 (기본: 100)')
    parser.add_argument('--lambda',      type=float, default=0.5, dest='lambda_',
                        help='Disagreement vs Coverage 균형 (0~1, 기본: 0.5)')
    parser.add_argument('--step-size',   type=float, default=0.01,
                        help='Gradient ascent step 크기 (기본: 0.01)')
    parser.add_argument('--output-dir',  type=str, default='results',
                        help='결과 저장 디렉토리 (기본: results/)')
    parser.add_argument('--force-retrain', action='store_true',
                        help='기존 모델 무시하고 재학습')
    args = parser.parse_args()

    print_banner()
    print(f"\n[설정]")
    print(f"  Device     : {DEVICE}")
    print(f"  Quick mode : {args.quick}")
    print(f"  Skip train : {args.skip_train}")
    print(f"  Output dir : {args.output_dir}/")

    # ── 1. 모델 준비 ──────────────────────────────────────
    print("\n[1단계] 모델 준비")
    if args.skip_train and not (os.path.exists('model_a.pth') and os.path.exists('model_b.pth')):
        print("  ※ 저장된 모델이 없습니다. --skip-train을 해제하거나 모델을 먼저 학습하세요.")
        return

    model_a, model_b = load_or_train_models(
        model_a_path='model_a.pth',
        model_b_path='model_b.pth',
        force_retrain=args.force_retrain,
        quick_train=args.quick
    )

    # 두 모델의 테스트 정확도 출력
    from models import get_cifar10_loaders, evaluate_model
    _, test_loader_eval = get_cifar10_loaders(batch_size=128)
    acc_a = evaluate_model(model_a, test_loader_eval)
    acc_b = evaluate_model(model_b, test_loader_eval)
    print(f"\n  Model A 테스트 정확도: {acc_a:.2f}%")
    print(f"  Model B 테스트 정확도: {acc_b:.2f}%")

    # ── 2. DeepXplore 실행 ───────────────────────────────
    print("\n[2단계] DeepXplore 실행")
    max_seeds = args.seeds if args.seeds is not None else (50 if args.quick else 200)

    print(f"  탐색 시드 수  : {max_seeds}")
    print(f"  최적화 스텝   : {args.steps}")
    print(f"  Lambda        : {args.lambda_}")
    print(f"  Step size     : {args.step_size}")

    # 시드 데이터로더 (batch_size=1로 하나씩 처리)
    test_loader_seed = get_test_loader(batch_size=1)

    dx = DeepXplore(
        model_a=model_a,
        model_b=model_b,
        coverage_threshold=0.5,
        lambda_=args.lambda_,
        step_size=args.step_size,
        num_steps=args.steps,
        perturbation_limit=0.1
    )

    results, coverage_a, coverage_b = dx.run(test_loader_seed, max_seeds=max_seeds)

    # ── 3. 결과 출력 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  실험 결과 요약")
    print("=" * 60)
    print(f"  탐색한 시드 수         : {max_seeds}")
    print(f"  Disagreement 발견 수   : {len(results)}")
    print(f"  Disagreement 비율      : {len(results)/max_seeds*100:.1f}%")
    print(f"  Model A Neuron Coverage: {coverage_a*100:.2f}%")
    print(f"  Model B Neuron Coverage: {coverage_b*100:.2f}%")
    print("=" * 60)

    if len(results) == 0:
        print("\n  Disagreement가 발견되지 않았습니다.")
        print("  --seeds 값을 늘리거나 --step-size를 조정해 보세요.")
        return

    # Disagreement 케이스 상세 출력 (상위 5개)
    print(f"\n  [상위 {min(5, len(results))}개 Disagreement 케이스]")
    for i, r in enumerate(results[:5]):
        true_cls  = CIFAR10_CLASSES[r['true_label']]
        pred_a_cls = CIFAR10_CLASSES[r['pred_a']]
        pred_b_cls = CIFAR10_CLASSES[r['pred_b']]
        print(f"  #{i+1}: True={true_cls:12s} | Model A={pred_a_cls:12s} | Model B={pred_b_cls:12s}")

    # ── 4. 시각화 저장 ───────────────────────────────────
    print(f"\n[3단계] 시각화 저장 → {args.output_dir}/")
    save_disagreement_plots(results, save_dir=args.output_dir, max_plots=10)
    save_summary_plot(results, coverage_a, coverage_b,
                      total_seeds=max_seeds, save_dir=args.output_dir)

    print(f"\n완료! 결과가 {args.output_dir}/ 디렉토리에 저장되었습니다.")


if __name__ == '__main__':
    main()
