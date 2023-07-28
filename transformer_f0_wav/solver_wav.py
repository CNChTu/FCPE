import numpy as np
import torch
import time
from transformer_f0_wav.saver.saver import Saver
from transformer_f0_wav.saver import utils
from torch import autocast
from torch.cuda.amp import GradScaler

from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy
from mir_eval.melody import voicing_recall, voicing_false_alarm


USE_MIR = True

def test(args, model, loader_test, saver):
    print(' [*] testing...')
    model.eval()

    # losses
    _rpa = _rca = _oa = _vfa = _vr = test_loss = 0.

    # intialization
    num_batches = len(loader_test)
    rtf_all = []

    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            f0 = model(mel=data['mel'], infer=True)
            ed_time = time.time()

            if USE_MIR:
                _f0 = f0.squeeze().cpu().numpy()
                _df0 = data['f0'].squeeze().cpu().numpy()

                # freq_pred = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in _f0])
                # freq = np.array([10 * (2 ** (cent / 1200)) if cent else 0 for cent in _df0])
                freq_pred = _f0
                freq = _df0
                time_slice = np.array([i * args.mel.hop_size / 1000 for i in range(len(_df0))])
                ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, freq, time_slice, freq_pred)

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
                loss = model(mel=data['mel'], infer=False, gt_f0=data['f0'])
                test_loss += loss.item()

            if USE_MIR:
                _rpa = _rpa + rpa
                _rca = _rca + rca
                _oa = _oa + oa
                _vfa = _vfa + vfa
                _vr = _vr + vr

            # log mel
            saver.log_spec(data['name'][0], data['mel'], data['mel'])

            saver.log_f0(data['name'][0], f0, data['f0'])
            saver.log_f0(data['name'][0], f0, data['f0'], inuv=True)

    # report
    test_loss /= args.train.batch_size
    test_loss /= num_batches

    if USE_MIR:
        _rpa /= args.train.batch_size
        _rpa /= num_batches

        _rca /= args.train.batch_size
        _rca /= num_batches

        _oa /= args.train.batch_size
        _oa /= num_batches

        _vfa /= args.train.batch_size
        _vfa /= num_batches

        _vr /= args.train.batch_size
        _vr /= num_batches

    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss, _rpa, _rca, _oa, _vfa, _vr


def train(args, initial_global_step, model, optimizer, scheduler, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    # run
    num_batches = len(loader_train)
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    for epoch in range(args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)

            # forward
            if dtype == torch.float32:
                loss = model(mel=data['mel'], infer=False, gt_f0=data['f0'])
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    loss = model(mel=data['mel'], infer=False, gt_f0=data['f0'])

            # handle nan loss
            if torch.isnan(loss):
                # raise ValueError(' [x] nan loss ')
                print(' [x] nan loss ')
                loss = None
                continue
            else:
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()

            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr = optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log / saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )

                saver.log_value({
                    'train/loss': loss.item()
                })

                saver.log_value({
                    'train/lr': current_lr
                })

            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None

                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')

                # run testing set
                test_loss, rpa, rca, oa, vfa, vr = test(args, model, loader_test, saver)

                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )

                saver.log_value({
                    'validation/loss': test_loss
                })
                if USE_MIR:
                    saver.log_value({
                        'validation/rpa': rpa
                    })
                    saver.log_value({
                        'validation/rca': rca
                    })
                    saver.log_value({
                        'validation/oa': oa
                    })
                    saver.log_value({
                        'validation/vfa': vfa
                    })
                    saver.log_value({
                        'validation/vr': vr
                    })

                model.train()
