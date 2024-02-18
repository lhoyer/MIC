from torch import irfft, rfft
import torch
import numpy as np

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:, :, :, :, 0]**2 + fft_im[:, :, :, :, 1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)  # get b
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]  # top left
    amp_src[:, :, 0:b, w - b:w] = amp_trg[:, :, 0:b, w - b:w]  # top right
    amp_src[:, :, h - b:h, 0:b] = amp_trg[:, :, h - b:h, 0:b]  # bottom left
    amp_src[:, :, h - b:h, w - b:w] = amp_trg[:, :, h - b:h,
                                              w - b:w]  # bottom right
    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = rfft(src_img.clone(), signal_ndim=2, onesided=False)
    fft_trg = rfft(trg_img.clone(), signal_ndim=2, onesided=False)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    fft_src_ = torch.zeros(
        fft_src.size(), dtype=torch.float, device=src_img.device)
    fft_src_[:, :, :, :, 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:, :, :, :, 1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = irfft(
        fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH, imgW])

    return src_in_trg