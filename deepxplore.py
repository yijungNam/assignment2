import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(DEVICE)
CIFAR10_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(DEVICE)

class NeuronCoverageTracker:
    def __init__(self, model: nn.Module, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
        self.coverage_dict = {}  # layer_name -> set of activated neuron indices
        self.total_neurons = {}  # layer_name -> total neuron count
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                self.coverage_dict[name] = set()
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, layer_name: str):
        def hook(module, input, output):
            activated = (output > self.threshold)
            # 배치 차원을 제거하고 뉴런 인덱스 기록
            if activated.dim() == 4:
                # Conv layer: (B, C, H, W) → C*H*W 개의 뉴런
                B, C, H, W = activated.shape
                flat = activated.view(B, -1)  
                total = C * H * W
            else:
                # FC layer: (B, C)
                flat = activated
                total = activated.shape[1]

            self.total_neurons[layer_name] = total
            # 어떤 배치 샘플에서든 한 번이라도 활성화된 뉴런 인덱스 저장
            newly_activated = flat.any(dim=0).nonzero(as_tuple=False).squeeze(1)
            self.coverage_dict[layer_name].update(newly_activated.cpu().tolist())
        return hook

    def get_coverage(self) -> float:
        total = sum(self.total_neurons.values())
        covered = sum(len(v) for v in self.coverage_dict.values())
        if total == 0:
            return 0.0
        return covered / total

    def reset(self):
        for key in self.coverage_dict:
            self.coverage_dict[key] = set()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class DeepXplore:
    def __init__(self, model_a: nn.Module, model_b: nn.Module,
                 coverage_threshold: float = 0.5,
                 lambda_: float = 0.5,
                 step_size: float = 0.01,
                 num_steps: int = 100,
                 perturbation_limit: float = 0.1):

        self.model_a = model_a.to(DEVICE).eval()
        self.model_b = model_b.to(DEVICE).eval()
        self.lambda_ = lambda_
        self.step_size = step_size
        self.num_steps = num_steps
        self.perturbation_limit = perturbation_limit

        self.tracker_a = NeuronCoverageTracker(model_a, threshold=coverage_threshold)
        self.tracker_b = NeuronCoverageTracker(model_b, threshold=coverage_threshold)

    def _disagreement_loss(self, out_a: torch.Tensor, out_b: torch.Tensor) -> torch.Tensor:
        prob_a = torch.softmax(out_a, dim=1)
        prob_b = torch.softmax(out_b, dim=1)

        pred_class = prob_a.argmax(dim=1)

        loss = prob_a[0, pred_class] - prob_b[0, pred_class]
        return loss

    def generate(self, seed_input: torch.Tensor):

        seed_input = seed_input.to(DEVICE)
        orig_input = seed_input.clone().detach()

        # perturbation을 학습 가능한 파라미터로 설정
        delta = torch.zeros_like(seed_input, requires_grad=True)

        best_input = seed_input.clone()
        best_disagreement = -float('inf')
        found_disagreement = False

        for step in range(self.num_steps):
            perturbed = seed_input + delta

            # L∞ 범위 내로 clamp
            perturbed = torch.clamp(
                perturbed,
                orig_input - self.perturbation_limit,
                orig_input + self.perturbation_limit
            )

            # 두 모델에 forward pass
            out_a = self.model_a(perturbed)
            out_b = self.model_b(perturbed)

            # Disagreement loss 계산
            dis_loss = self._disagreement_loss(out_a, out_b)

            prob_a = torch.softmax(out_a, dim=1)
            entropy = -(prob_a * (prob_a + 1e-8).log()).sum()
            cov_loss = entropy

    
            total_loss = self.lambda_ * dis_loss + (1 - self.lambda_) * cov_loss

            if delta.grad is not None:
                delta.grad.zero_()
            total_loss.backward()

            with torch.no_grad():
                delta.data += self.step_size * delta.grad.sign()

                # 현재 불일치 강도 기록
                pred_a = out_a.argmax(dim=1).item()
                pred_b = out_b.argmax(dim=1).item()

                if pred_a != pred_b:
                    found_disagreement = True
                    dis_val = dis_loss.item()
                    if dis_val > best_disagreement:
                        best_disagreement = dis_val
                        best_input = perturbed.detach().clone()

        with torch.no_grad():
            final_input = (seed_input + delta).clamp(
                orig_input - self.perturbation_limit,
                orig_input + self.perturbation_limit
            )
            final_out_a = self.model_a(final_input)
            final_out_b = self.model_b(final_input)
            pred_a = final_out_a.argmax(dim=1).item()
            pred_b = final_out_b.argmax(dim=1).item()

        if not found_disagreement and pred_a == pred_b:
            return final_input.detach(), pred_a, pred_b, False
        else:
            if not found_disagreement:
                best_input = final_input.detach()
            return best_input, pred_a, pred_b, (pred_a != pred_b)

    def run(self, data_loader, max_seeds: int = 200):
        results = []
        seed_count = 0

        print(f"\n[DeepXplore] {max_seeds}개의 시드 입력 탐색 시작")

        for batch_inputs, batch_labels in tqdm(data_loader, desc="DeepXplore 실행", total = max_seeds ):
            for i in range(batch_inputs.size(0)):
                if seed_count >= max_seeds:
                    break

                seed = batch_inputs[i:i+1].to(DEVICE)
                true_label = batch_labels[i].item()

                with torch.no_grad():
                    self.model_a(seed)
                    self.model_b(seed)

                gen_input, pred_a, pred_b, found = self.generate(seed)

                if found:
                    results.append({
                        'input': gen_input.cpu(),
                        'seed': seed.cpu(),
                        'true_label': true_label,
                        'pred_a': pred_a,
                        'pred_b': pred_b,
                    })

                seed_count += 1

            if seed_count >= max_seeds:
                break

        coverage_a = self.tracker_a.get_coverage()
        coverage_b = self.tracker_b.get_coverage()

        print(f"\n[결과] 탐색한 시드 수: {seed_count}")
        print(f"[결과] Disagreement 발견: {len(results)}개")
        print(f"[결과] Model A Neuron Coverage: {coverage_a*100:.2f}%")
        print(f"[결과] Model B Neuron Coverage: {coverage_b*100:.2f}%")

        self.tracker_a.remove_hooks()
        self.tracker_b.remove_hooks()

        return results, coverage_a, coverage_b
