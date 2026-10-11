"""Exercise CUDA and a pinned SpUNet on real geometry with synthetic labels.

This is an environment check, not weld training or an accuracy evaluation.
Run through training/check_env.ps1 so project-local caches are configured.
"""
import argparse
from datetime import datetime
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "training/reports/environment_check.json")
    args = parser.parse_args()
    report = {
        "started_at": datetime.now().isoformat(), "status": "RUNNING",
        "scope": "Environment only; real XYZ crop, synthetic random targets; no weld accuracy claim.",
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "requested_points": args.points, "requested_steps": args.steps,
        "small_variant": args.small, "checks": {},
    }
    code = 1
    try:
        import numpy as np
        import torch
        import spconv.pytorch as spconv
        import open3d as o3d
        from scipy.spatial import cKDTree
        from training.models.spunet import SpUNetBase
        from training.dataio import read_xyz_pcd

        packages = ["torch", "spconv-cu116", "cumm-cu116", "numpy", "scipy", "open3d",
                    "pandas", "openpyxl", "plyfile", "PyYAML", "tensorboard", "scikit-learn"]
        report["packages"] = {p: importlib.metadata.version(p) for p in packages}
        report["gpu_query"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
            text=True,
        ).strip()
        assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
        report["cuda_runtime"] = torch.version.cuda
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_capability"] = torch.cuda.get_device_capability(0)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        torch.set_num_threads(4)
        a = torch.randn(256, 256, device="cuda", requires_grad=True)
        (a @ a.T).square().mean().backward()
        assert bool(torch.isfinite(a.grad).all())
        report["checks"]["cuda_matmul_backward"] = "PASS"
        print("PASS CUDA matmul/backward", flush=True)
        del a

        indices = torch.randint(0, 48, (4096, 3), device="cuda", dtype=torch.int32).unique(dim=0)
        indices = torch.cat([torch.zeros_like(indices[:, :1]), indices], dim=1)
        features = torch.randn(len(indices), 3, device="cuda", requires_grad=True)
        conv = spconv.SubMConv3d(3, 8, 3, padding=1, indice_key="environment").cuda()
        sparse = spconv.SparseConvTensor(features, indices, [48, 48, 48], 1)
        conv(sparse).features.float().square().mean().backward()
        assert bool(torch.isfinite(features.grad).all())
        report["checks"]["spconv_backward"] = "PASS"
        print("PASS sparse convolution/backward", flush=True)
        del indices, features, conv, sparse
        gc.collect()
        torch.cuda.empty_cache()

        files = list((ROOT / "DataSet/9月20日").glob("*/*_AllFaces.pcd"))
        source = max(files, key=lambda p: p.stat().st_size)
        xyz = read_xyz_pcd(source) * 1000.0
        assert xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all()
        report["source_file"] = str(source.relative_to(ROOT))
        report["source_points"] = len(xyz)
        report["checks"]["unicode_ascii_pcd_read"] = "PASS"
        # Scaling is the audited working hypothesis, not a PCD-declared unit.
        report["pcd_scale_to_mm"] = 1000
        report["voxel_mm"] = 2.0
        center = xyz[len(xyz) // 2]
        _, ix = cKDTree(xyz).query(center, k=min(len(xyz), args.points * 3), workers=1)
        crop = xyz[ix]
        grid = np.floor((crop - crop.min(axis=0)) / report["voxel_mm"]).astype(np.int32)
        _, first = np.unique(grid, axis=0, return_index=True)
        first = np.sort(first)[:args.points]
        crop, grid = crop[first], grid[first]
        grid -= grid.min(axis=0)
        feat = ((crop - crop.mean(axis=0)) / 1000.0).astype(np.float32)
        report["actual_points"] = len(feat)
        report["crop_extent_mm"] = np.ptp(crop, axis=0).tolist()
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(crop[:1024]))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=16))
        assert np.isfinite(np.asarray(cloud.normals)).all()
        report["checks"]["open3d_normal_estimation"] = "PASS"
        del cloud, xyz, crop, ix, first
        data = {
            "grid_coord": torch.from_numpy(grid).cuda(),
            "feat": torch.from_numpy(feat).cuda(),
            "offset": torch.tensor([len(feat)], device="cuda", dtype=torch.long),
        }
        config = {"in_channels": 3, "num_classes": 2}
        if args.small:
            config.update(base_channels=16, channels=(16, 32, 64, 128, 128, 64, 48, 48))
        model = SpUNetBase(**config).cuda().train()
        report["model_config"] = config
        report["parameters"] = sum(p.numel() for p in model.parameters())
        target = torch.randint(0, 2, (len(feat),), device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler(init_scale=1024)
        before = model.final.weight.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        durations, losses = [], []
        for step in range(args.steps):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(data)
                assert tuple(logits.shape) == (len(feat), 2)
                loss = torch.nn.functional.cross_entropy(logits.float(), target)
            assert bool(torch.isfinite(loss))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            losses.append(float(loss.detach()))
            print(f"SpUNet step {step+1}/{args.steps}: points={len(feat)}, loss={losses[-1]:.6f}, seconds={durations[-1]:.3f}", flush=True)
        assert not torch.equal(before, model.final.weight.detach()), "Optimizer did not update the final layer"
        report.update(
            step_seconds=durations, synthetic_target_losses=losses,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
            amp=True,
        )
        report["checks"]["spunet_forward_backward_optimizer"] = "PASS"
        report["status"] = "PASS"
        code = 0
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report["finished_at"] = datetime.now().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{report['status']}: {args.output}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
