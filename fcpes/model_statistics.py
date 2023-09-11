import argparse
import torch
import time
from torch.optim import lr_scheduler
from transformer_f0_wav.saver import utils
from transformer_f0_wav.data_loaders_wav import get_data_loaders
from transformer_f0_wav.model_with_bce import TransformerF0BCE
from transformer_f0_wav.model_with_bce import Wav2Mel
from transformer_f0_wav.solver_wav import train
from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy
from mir_eval.melody import voicing_recall, voicing_false_alarm
import transformer_f0_wav.utils as ut
from data_loaders_wav import F0Dataset
import numpy as np

USE_MIR = True

def test(args, model, loader_test):
    print(' [*] testing...')
    model.eval()

    # losses
    _rpa = _rca = _oa = _vfa = _vr = test_loss = 0.
    _num_a = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []

    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data[2][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            #for k in data.keys():
            #    if not k.startswith('name'):
            #        data[k] = data[k].to(args.device)
            for k in range(len(data)):
                if k < 2:
                    data[k] = data[k].to(args.device)
            #print('>>', data[2][0])
            # forward
            st_time = time.time()
            f0 = model(mel=data[0], infer=True)
            ed_time = time.time()

            if USE_MIR:
                _f0 = ((f0.exp() - 1) * 700).squeeze().cpu().numpy()
                _df0 = data[1].squeeze().cpu().numpy()

                time_slice = np.array([i * args.mel.hop_size * 1000 / args.mel.sampling_rate for i in range(len(_df0))])
                ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, _df0, time_slice, _f0)

                rpa = raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
                rca = raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
                oa = overall_accuracy(ref_v, ref_c, est_v, est_c)
                vfa = voicing_false_alarm(ref_v, est_v)
                vr = voicing_recall(ref_v, est_v)

            # RTF
            run_time = ed_time - st_time
            song_time = f0.shape[1] * args.mel.hop_size / args.mel.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            if USE_MIR:
                print('RPA: {}  | RCA: {} | OA: {} | VFA: {} | VR: {} |'.format(rpa, rca, oa, vfa, vr))
            rtf_all.append(rtf)

            # loss
            for i in range(args.train.batch_size):
                loss = model(mel=data[0], infer=False, gt_f0=data[1])
                test_loss += loss.item()

            if USE_MIR:
                _rpa = _rpa + rpa
                _rca = _rca + rca
                _oa = _oa + oa
                _vfa = _vfa + vfa
                _vr = _vr + vr
                _num_a = _num_a + 1

    # report
    test_loss /= args.train.batch_size
    test_loss /= num_batches

    if USE_MIR:
        _rpa /= _num_a

        _rca /= _num_a

        _oa /= _num_a

        _vfa /= _num_a

        _vr /= _num_a

    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss, _rpa, _rca, _oa, _vfa, _vr


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    
    parser.add_argument(
        "-snb",
        "--noise_snb",
        type=float,
        default= 0,
        required=True,
        help="noise snb")
    
    parser.add_argument(
        "-b",
        "--beta",
        type=int,
        default= 0,
        required=True,
        help="beta")

    return parser.parse_args(args=args, namespace=namespace)


cmd = parse_args()

args = utils.load_config(cmd.config)

model = TransformerF0BCE(
            input_channel=args.model.input_channel,
            out_dims=args.model.out_dims,
            n_layers=args.model.n_layers,
            n_chans=args.model.n_chans,
            use_siren=args.model.use_siren,
            use_full=args.model.use_full,
            loss_mse_scale=args.loss.loss_mse_scale,
            loss_l2_regularization=args.loss.loss_l2_regularization,
            loss_l2_regularization_scale=args.loss.loss_l2_regularization_scale,
            loss_grad1_mse=args.loss.loss_grad1_mse,
            loss_grad1_mse_scale=args.loss.loss_grad1_mse_scale,
            f0_max=args.model.f0_max,
            f0_min=args.model.f0_min,
            confidence=args.model.confidence,
            threshold=args.model.threshold,
            use_input_conv=args.model.use_input_conv,
            residual_dropout=args.model.residual_dropout,
            attention_dropout=args.model.attention_dropout,
            cdecoder=args.model.cdecoder
        )

initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, None, device=args.device)

wav2mel = Wav2Mel(args, device='cpu')

data_valid = F0Dataset(
        path_root=args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.mel.hop_size,
        sample_rate=args.mel.sampling_rate,
        duration=args.data.duration,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        wav2mel=wav2mel,
        aug_noise=args.train.aug_noise,
        noise_ratio=args.train.noise_ratio,
        brown_noise_ratio=args.train.brown_noise_ratio,
        aug_flip=False,
        aug_mask=False,
        aug_mask_v_o=args.train.aug_mask_v_o,
        aug_mask_vertical_factor=args.train.aug_mask_vertical_factor,
        aug_mask_vertical_factor_v_o=False,
        aug_mask_iszeropad_mode=args.train.aug_mask_iszeropad_mode,
        aug_mask_block_num=args.train.aug_mask_block_num,
        aug_mask_block_num_v_o=args.train.aug_mask_block_num_v_o,
        aug_eq=False,
        aug_keyshift=False,
        aug_reverb=False,
        snb_noise=cmd.noise_snb,
        noise_beta= cmd.beta
    )

loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


test(args, model, loader_valid)