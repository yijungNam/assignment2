import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CIFAR-10 역정규화 파라미터
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD  = np.array([0.2023, 0.1994, 0.2010])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.squeeze().cpu().numpy()  # (3, H, W)
    img = img.transpose(1, 2, 0)           # (H, W, 3)
    img = img * STD + MEAN                 # 역정규화
    img = np.clip(img, 0, 1)
    return img


def save_disagreement_plots(results: list, save_dir: str = 'results', max_plots: int = 10):
    """
        results: deepxplore.run()이 반환한 결과 리스트
        save_dir: 저장 디렉토리
        max_plots: 저장할 최대 케이스 수
    """
    os.makedirs(save_dir, exist_ok=True)

    n = min(len(results), max_plots)
    print(f"\n[시각화] {n}개의 disagreement 케이스 저장 중...")

    for idx in range(n):
        r = results[idx]
        seed_img  = denormalize(r['seed'])
        gen_img   = denormalize(r['input'])
        true_lbl  = CIFAR10_CLASSES[r['true_label']]
        pred_a    = CIFAR10_CLASSES[r['pred_a']]
        pred_b    = CIFAR10_CLASSES[r['pred_b']]

        # Perturbation 시각화 
        diff = np.abs(gen_img - seed_img)
        diff_amplified = np.clip(diff * 5, 0, 1)  # 5배 증폭

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        fig.suptitle(
            f"Case #{idx+1} | True Label: {true_lbl}\n"
            f"Model A: {pred_a}  |  Model B: {pred_b}",
            fontsize=11, fontweight='bold'
        )

        axes[0].imshow(seed_img)
        axes[0].set_title("Seed Input", fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(gen_img)
        axes[1].set_title(f"Generated Input\n(A→{pred_a} / B→{pred_b})", fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(diff_amplified)
        axes[2].set_title("Perturbation (×5)", fontsize=10)
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'disagreement_{idx+1:03d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  저장: {save_path}")


def save_summary_plot(results: list, coverage_a: float, coverage_b: float,
                      total_seeds: int, save_dir: str = 'results'):
    """
    전체 실험 결과 요약 그래프 저장.
    - 상위 5개 disagreement 케이스 이미지
    - Coverage 막대 그래프
    - Disagreement 클래스 분포
    """
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("DeepXplore 실험 결과 요약", fontsize=14, fontweight='bold', y=1.01)

    n_show = min(5, len(results))
    gs_top = gridspec.GridSpec(1, n_show, top=1.0, bottom=0.6, hspace=0.4, wspace=0.3)

    for i in range(n_show):
        r = results[i]
        ax = fig.add_subplot(gs_top[0, i])
        ax.imshow(denormalize(r['input']))
        ax.set_title(
            f"A: {CIFAR10_CLASSES[r['pred_a']]}\nB: {CIFAR10_CLASSES[r['pred_b']]}",
            fontsize=8
        )
        ax.axis('off')

    ax_cov = fig.add_subplot(gridspec.GridSpec(1, 2, top=0.5, bottom=0.05)[0, 0])
    bars = ax_cov.bar(['Model A', 'Model B'],
                      [coverage_a * 100, coverage_b * 100],
                      color=['steelblue', 'coral'], edgecolor='black', width=0.4)
    for bar, val in zip(bars, [coverage_a * 100, coverage_b * 100]):
        ax_cov.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    ax_cov.set_ylim(0, 110)
    ax_cov.set_ylabel('Neuron Coverage (%)')
    ax_cov.set_title('Neuron Coverage 비교')
    ax_cov.set_yticks(range(0, 110, 10))

    ax_dist = fig.add_subplot(gridspec.GridSpec(1, 2, top=0.5, bottom=0.05)[0, 1])

    from collections import Counter
    true_labels = [CIFAR10_CLASSES[r['true_label']] for r in results]
    label_counts = Counter(true_labels)

    if label_counts:
        classes, counts = zip(*sorted(label_counts.items(), key=lambda x: -x[1]))
        ax_dist.bar(classes, counts, color='mediumseagreen', edgecolor='black')
        ax_dist.set_xlabel('True Class')
        ax_dist.set_ylabel('Disagreement 발생 횟수')
        ax_dist.set_title('클래스별 Disagreement 분포')
        plt.setp(ax_dist.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    fig.text(0.5, 0.55,
             f"총 시드 수: {total_seeds}  |  Disagreement 발견: {len(results)}개  "
             f"|  Disagreement Rate: {len(results)/max(total_seeds,1)*100:.1f}%",
             ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'summary.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n[시각화] 요약 그래프 저장: {save_path}")
