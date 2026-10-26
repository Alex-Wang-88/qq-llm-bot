"""Fail-closed GPU numerical/autograd/AdamW acceptance test, not an LLM result."""
import argparse
import copy
import json
import time
import traceback
from common import LOCAL, configure, memory, save_json, stamp


def check_dtype(torch, dtype):
    torch.manual_seed(3407)
    base = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.SiLU(), torch.nn.Linear(64, 8))
    x = torch.randn(32, 32)
    target = torch.randn(32, 8)
    # CPU reference starts from exactly the same rounded inputs/weights.
    gpu = copy.deepcopy(base).to(device='cuda', dtype=dtype)
    cpu = copy.deepcopy(gpu).to(device='cpu', dtype=torch.float32)
    rounded_x = x.to(dtype).float()
    reference = cpu(rounded_x)
    result = gpu(x.to(device='cuda', dtype=dtype))
    torch.cuda.synchronize()
    error = (result.detach().float().cpu() - reference.detach()).abs()
    relative_l2 = float(error.norm() / reference.detach().norm().clamp_min(1e-8))
    gpu_loss = (result.float() - target.cuda()).square().mean()
    cpu_loss = (reference - target).square().mean()
    gpu_loss.backward()
    cpu_loss.backward()
    gradient_errors = []
    for gp, cp in zip(gpu.parameters(), cpu.parameters()):
        assert gp.grad is not None and torch.isfinite(gp.grad).all()
        gradient_errors.append(float((gp.grad.float().cpu() - cp.grad).norm() / cp.grad.norm().clamp_min(1e-8)))
    threshold = {torch.float32: 0.002, torch.float16: 0.03, torch.bfloat16: 0.12}[dtype]
    assert relative_l2 < threshold, f'Forward error {relative_l2}'
    assert max(gradient_errors) < threshold, f'Gradient errors {gradient_errors}'
    # Real LoRA uses FP32 adapter/master parameters. Native FP16 optimizer state
    # is deliberately avoided: tiny squared gradients can underflow.
    gpu = gpu.float()
    gpu.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(gpu.parameters(), lr=0.004, eps=1e-4, foreach=False, fused=False)
    before = [p.detach().clone() for p in gpu.parameters()]
    losses = []
    started = time.perf_counter()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
            loss = (gpu(x.cuda()).float() - target.cuda()).square().mean()
        assert torch.isfinite(loss), 'Non-finite loss'
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in gpu.parameters()), 'Non-finite gradient'
        if not losses:
            cpu_update = copy.deepcopy(gpu).cpu()
            for cp, gp in zip(cpu_update.parameters(), gpu.parameters()): cp.grad = gp.grad.detach().cpu().clone()
            cpu_optimizer = torch.optim.AdamW(cpu_update.parameters(), lr=0.004, eps=1e-4, foreach=False, fused=False)
            cpu_optimizer.step()
        optimizer.step()
        if not losses:
            update_error = max(float((gp.detach().cpu()-cp.detach()).abs().max()) for gp,cp in zip(gpu.parameters(),cpu_update.parameters()))
            assert update_error < 1e-5, f'AdamW CPU/GPU disagreement: {update_error}'
        assert all(torch.isfinite(p).all() for p in gpu.parameters()), 'Non-finite parameter'
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    deltas = [float((a - b).float().norm().cpu()) for a, b in zip(gpu.parameters(), before)]
    assert all(d > 0 for d in deltas), 'Parameters were not updated'
    assert losses[-1] < losses[0] * .9, 'No meaningful optimization'
    return {'dtype': str(dtype), 'passed': True, 'forward_relative_l2': relative_l2,
            'forward_max_absolute': float(error.max()), 'gradient_relative_l2': gradient_errors,
            'parameter_delta_l2': deltas, 'optimizer_parameter_dtype':'float32', 'optimizer_cpu_gpu_max_abs':update_error,
            'losses': losses, 'seconds': time.perf_counter()-started,
            'memory': memory(torch)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(LOCAL / 'gpu-smoke.json'))
    args = parser.parse_args()
    report = {'started': stamp(), 'passed': False, 'tests': []}
    try:
        torch = configure()
        report.update(torch_version=torch.__version__, hip_version=torch.version.hip,
                      cuda_version=torch.version.cuda, cuda_available=torch.cuda.is_available())
        assert torch.version.hip, 'This is not an AMD HIP build'
        assert torch.cuda.is_available(), 'No GPU recognized'
        report['devices'] = [str(torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())]
        assert '9070' in torch.cuda.get_device_name(0), 'Expected discrete RX 9070 XT at device 0'
        a = torch.arange(64, device='cuda', dtype=torch.float32).reshape(8, 8)
        mm = a @ a.T
        torch.testing.assert_close(mm.cpu(), a.cpu() @ a.cpu().T, rtol=1e-5, atol=1e-4)
        report['matmul_passed'] = True
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            try:
                row = check_dtype(torch, dtype)
            except Exception:
                row = {'dtype': str(dtype), 'passed': False, 'error': traceback.format_exc()}
            report['tests'].append(row)
            save_json(args.output, report)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report['passed'] = all(t['passed'] for t in report['tests'])
    except Exception:
        report['error'] = traceback.format_exc()
    report['finished'] = stamp()
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
