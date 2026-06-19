"""Minimal checkpoint verification for the incremental-learning Maxwell model.

This script loads a fine-tuned checkpoint, runs prediction on a small regular
(x, y, t) grid, and prints numeric summaries. It does not require FDTD labels.
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import mindspore.common.dtype as ms_type
from mindspore import Tensor, Parameter, context
from mindspore.common.initializer import HeUniform
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindelec.architecture import MultiScaleFCCell


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remap_checkpoint_params(raw_params):
    """Map Solver checkpoint names to the standalone MultiScaleFCCell names."""
    param_dict = {}
    for name, param in raw_params.items():
        if name == "model.latent_vector":
            param_dict["latent_vector"] = param
            continue
        matched = re.match(r"model\.cell_list\.(\d+)\.network\.(.+)", name)
        if not matched:
            continue
        idx = int(matched.group(1))
        rest = matched.group(2)
        if idx == 0:
            new_name = "0.network." + rest
        else:
            new_name = f"0.network.{idx}.network.{rest}"
        param_dict[new_name] = param
    return param_dict


def build_network(config, raw_params):
    latent = raw_params["model.latent_vector"].data.asnumpy().astype(np.float32)
    latent_vector = Parameter(Tensor(latent, ms_type.float32), requires_grad=False)
    network = MultiScaleFCCell(
        config["input_size"],
        config["output_size"],
        layers=config["layers"],
        neurons=config["neurons"],
        residual=config["residual"],
        weight_init=HeUniform(negative_slope=math.sqrt(5)),
        act="sin",
        num_scales=config["num_scales"],
        amp_factor=config["amp_factor"],
        scale_factor=config["scale_factor"],
        input_scale=config["input_scale"],
        input_center=config["input_center"],
        latent_vector=latent_vector,
    )
    network = network.to_float(ms_type.float16)
    network.input_scale.to_float(ms_type.float32)
    not_loaded = load_param_into_net(network, remap_checkpoint_params(raw_params))
    network.set_train(False)
    return network, not_loaded


def predict_grid(network, config, nx, ny, nt, batch_size):
    xs = np.linspace(config["coord_min"][0], config["coord_max"][0], nx, dtype=np.float32)
    ys = np.linspace(config["coord_min"][1], config["coord_max"][1], ny, dtype=np.float32)
    ts = np.linspace(0.0, config["range_t"], nt, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    output_scale = np.array(config["output_scale"], dtype=np.float32)
    pred = np.zeros((nt, nx, ny, config["output_size"]), dtype=np.float32)
    for ti, t in enumerate(ts):
        coords = np.stack(
            [xx.reshape(-1), yy.reshape(-1), np.full(nx * ny, t, dtype=np.float32)],
            axis=1,
        )
        parts = []
        for start in range(0, len(coords), batch_size):
            batch = Tensor(coords[start : start + batch_size], ms_type.float32)
            parts.append(network(batch).asnumpy().astype(np.float32) * output_scale)
        pred[ti] = np.concatenate(parts, axis=0).reshape(nx, ny, config["output_size"])
    return pred


def main():
    parser = argparse.ArgumentParser(description="Verify a fine-tuned Maxwell checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--config", default="config/reconstruct.json", help="Path to reconstruct config")
    parser.add_argument("--device-target", default="Ascend", choices=["Ascend", "CPU", "GPU"])
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nt", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=args.device_target)
    config = load_config(args.config)
    raw_params = load_checkpoint(str(Path(args.ckpt)))
    network, not_loaded = build_network(config, raw_params)
    pred = predict_grid(network, config, args.nx, args.ny, args.nt, args.batch_size)

    names = ["Ex", "Ey", "Hz"]
    print("checkpoint:", args.ckpt)
    print("config:", args.config)
    print("grid_shape:", pred.shape)
    print("not_loaded:", not_loaded)
    for i, name in enumerate(names):
        values = pred[..., i]
        print(
            f"{name}: min={values.min():.9g}, max={values.max():.9g}, "
            f"mean_abs={np.mean(np.abs(values)):.9g}, rmse={np.sqrt(np.mean(values ** 2)):.9g}"
        )
    ic = np.abs(pred[0])
    for i, name in enumerate(names):
        print(f"IC |{name}(t=0)| mean={ic[..., i].mean():.9g}, max={ic[..., i].max():.9g}")


if __name__ == "__main__":
    main()
