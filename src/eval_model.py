import argparse
from collections import defaultdict
from tqdm import tqdm
import torch
import time
from utils import os_utils
import numpy as np
from test_dataset import TestDataset
from src.model_with_bce import TransformerF0BCE, Wav2Mel, DotDict
from mir_eval.melody import (
    # acc
    raw_pitch_accuracy,
    raw_chroma_accuracy,
    overall_accuracy,
    # uv
    voicing_recall,
    voicing_false_alarm,
    # utils
    to_cent_voicing,
)

EVAL = True


def main(model: TransformerF0BCE, test_dataset: TestDataset, args: DotDict):
    model.eval()

    total_rtf = 0
    total_stats = defaultdict(float)

    with torch.no_grad():
        for idx, test_data in enumerate(tqdm(test_dataset)):
            mel, gt_f0, _, _ = test_data

            mel = mel.unsqueeze(0).to(args.device)
            gt_f0 = gt_f0.to(args.device)

            st_time = time.time()
            f0_hat = model.forward(mel, infer=True, return_hz_f0=True)
            ed_time = time.time()

            if EVAL:
                f0_hat = f0_hat.squeeze().cpu().numpy()
                gt_f0 = gt_f0.squeeze().cpu().numpy()

                time_slice_ms = (
                    np.arange(len(gt_f0))
                    * args.mel.hop_size
                    / args.mel.sampling_rate
                    * 1000
                )
                ref_voicing, ref_cent, est_voicing, est_cent = to_cent_voicing(
                    time_slice_ms, gt_f0, time_slice_ms, f0_hat
                )

                stats = {
                    "rpa": raw_pitch_accuracy(
                        ref_voicing, ref_cent, est_voicing, est_cent
                    ),
                    "rca": raw_chroma_accuracy(
                        ref_voicing, ref_cent, est_voicing, est_cent
                    ),
                    "oa": overall_accuracy(
                        ref_voicing, ref_cent, est_voicing, est_cent
                    ),
                    "vfa": voicing_false_alarm(ref_voicing, est_voicing),
                    "vr": voicing_recall(ref_voicing, est_voicing),
                }

                print(" |".join([f"{k}: {v}" for k, v in stats.items()]))
                for k, v in stats.items():
                    total_stats[k] += v

            if idx > 0:  # first time takes longer
                run_time = ed_time - st_time
                song_time = len(f0_hat) * args.mel.hop_size / args.mel.sampling_rate
                rtf = run_time / song_time

                print(f"RTF: {rtf}  | {run_time} / {song_time}")
                total_rtf += rtf

    dataset_size = len(test_dataset)
    mean_rtf = total_rtf / (dataset_size - 1)  # does not count the first time
    if EVAL:
        mean_stats = {}
        for k, v in total_stats.items():
            mean_stats[k] = total_stats[k] / dataset_size

    print(" Real Time Factor", mean_rtf)
    return mean_stats


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to the config file"
    )
    parser.add_argument(
        "-snb", "--noise_snb", type=float, default=0, required=True, help="noise snb"
    )
    parser.add_argument("-b", "--beta", type=int, default=0, required=True, help="beta")

    return parser.parse_args(args=args, namespace=namespace)


if __name__ == "__main__":
    cmd = parse_args()
    args = os_utils.load_config(cmd.config)

    model = TransformerF0BCE(
        # model hyperparameter
        f0_min=args.model.f0_min,
        f0_max=args.model.f0_max,
        threshold=args.model.threshold,
        # model architecture
        use_siren=args.model.use_siren,
        use_full=args.model.use_full,
        use_input_conv=args.model.use_input_conv,
        residual_dropout=args.model.residual_dropout,
        attention_dropout=args.model.attention_dropout,
        cdecoder=args.model.cdecoder,
        # model size definition
        out_dims=args.model.out_dims,
        input_channel=args.model.input_channel,
        n_chans=args.model.n_chans,
        n_layers=args.model.n_layers,
    )

    _, model, _ = os_utils.load_model(args.env.expdir, model, None, device=args.device)

    test_dataset = TestDataset(
        # common settings
        path_root=args.data.valid_path,
        extensions=args.data.extensions,
        sample_rate=args.mel.sampling_rate,
        wav2mel=Wav2Mel(args, device="cpu"),
        whole_audio=True,
        load_all_data=False,
        # eval protocol
        hop_size=args.mel.hop_size,
        # distortions
        noise_ratio=args.train.noise_ratio,
        brown_noise_ratio=args.train.brown_noise_ratio,
        snb_noise=cmd.noise_snb,
        noise_beta=cmd.beta,
    )

    main(model.to(args.device), test_dataset, args)
