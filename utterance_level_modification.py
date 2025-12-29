import os
import soundfile as sf
import pyworld as pw
import numpy as np
import math
import scipy.interpolate
import argparse
import yaml
from tqdm import tqdm

def read_phn(file):
    # Read phoneme alignment file and return vowel segments
    with open(file) as f:
        lines = f.readlines()
        ali_phn = [l.strip().split() for l in lines]
        ali_phn = [l for l in ali_phn if l[2][0] in ["a", "e", "i", "o", "u"]]
        ali_phn = [[int(l[0]), int(l[1]), l[2]] for l in ali_phn]
    return ali_phn

def make_intensity_mask(log_energy, intensity_scale_by_vowel, f0, ali_phn, fs, frame_period):
    # Create a mask for intensity scaling based on vowel segments or voiced frames
    if intensity_scale_by_vowel:
        mask = np.zeros_like(log_energy, dtype=bool)
        for ali in ali_phn:
            start_frame = int(math.floor(int(ali[0])) / (fs * frame_period / 1000))
            end_frame = int(math.ceil(int(ali[1])) / (fs * frame_period / 1000))
            mask[start_frame:end_frame] = True
    else:
        mask = f0 > 0
    return mask

def scale_spectral_envelope(sp, rate):
    # Scale the spectral envelope by the given rate for vocal tract length modification
    n_frames, fft_size = sp.shape
    original_freq = np.linspace(0, 1, fft_size)
    scaled_freq = np.linspace(0, 1, int(fft_size / rate))

    scaled_sp = []
    for frame in sp:
        interp_func = scipy.interpolate.interp1d(original_freq, frame, kind='linear', fill_value="extrapolate")
        new_frame = interp_func(scaled_freq)
        if len(new_frame) < fft_size:
            new_frame = np.pad(new_frame, (0, fft_size - len(new_frame)), mode='edge')
        else:
            new_frame = new_frame[:fft_size]
        scaled_sp.append(new_frame)
    
    return np.array(scaled_sp)

def main(config):
    frame_period = config["frame_period"]
    pitch_scale = config["pitch_scale"]
    intensity_scale = config["intensity_scale"]
    intensity_scale_by_vowel = config["intensity_scale_by_vowel"]

    for script in tqdm(os.listdir("SX")):
        for dr in os.listdir(os.path.join("SX", script)):
            for spk in os.listdir(os.path.join("SX", script, dr)):
                os.makedirs(os.path.join("SX", script, dr, spk, "synth_utt"), exist_ok=True)
                for file in os.listdir(os.path.join("SX", script, dr, spk, "synth_utt")):
                    os.system(f"rm {os.path.join('SX', script, dr, spk, 'synth_utt', file)}")
                bname = f"{dr}_{spk}"
                os.system(f"cp {os.path.join('SX', script, dr, spk, f'{script}.wav')} {os.path.join('SX', script, dr, spk, 'synth_utt', f'{bname}-{script}.wav')}")

                orig_wav = os.path.join("SX", script, dr, spk, f"{script}.wav")
                x, fs = sf.read(orig_wav)
                orig_phn = os.path.join("SX", script, dr, spk, f"{script}.phn")
                ali_phn = read_phn(orig_phn)

                f0, sp, ap = pw.wav2world(x, fs, frame_period=frame_period)
                
                resyn = pw.synthesize(f0, sp, ap, fs, frame_period=frame_period)
                sf.write(os.path.join("SX", script, dr, spk, "synth_utt", f"{bname}-{script}_resyn.wav"), resyn, fs)

                # pitch scale
                for alpha in pitch_scale:
                    mean_f0 = np.mean(f0[f0 > 0])
                    new_f0 = np.where(f0 > 0, (f0 - mean_f0) * alpha + mean_f0, f0)
                    scaled = pw.synthesize(
                        new_f0, sp, ap, fs, frame_period=frame_period
                    )
                    sf.write(
                        os.path.join("SX", script, dr, spk, "synth_utt", f"{bname}-{script}_pitch_scale_{alpha}.wav"),
                        scaled,
                        fs,
                    )

                # intensity scale
                energy = np.mean(sp, axis=1)
                log_energy = np.log(energy + 1e-10)
                mask = make_intensity_mask(log_energy, intensity_scale_by_vowel, f0, ali_phn, fs, frame_period)
                mean_log_energy = np.mean(log_energy[mask])
                for alpha in intensity_scale:
                    new_log_energy = np.where(mask, (log_energy - mean_log_energy) * alpha + mean_log_energy, log_energy)
                    gain = np.where(mask, np.exp(new_log_energy - log_energy), 1)
                    scaled = pw.synthesize(
                        f0, sp * gain[:, np.newaxis], ap, fs, frame_period=frame_period
                    )
                    sf.write(
                        os.path.join("SX", script, dr, spk, "synth_utt", f"{bname}-{script}_intensity_scale_{alpha}.wav"),
                        scaled,
                        fs,
                    )
                # voice change(vocal tract length)
                for alpha in config["vocal_tract_length"]:
                    new_sp = scale_spectral_envelope(sp, alpha)
                    scaled = pw.synthesize(
                        f0, new_sp, ap, fs, frame_period=frame_period
                    )
                    sf.write(
                        os.path.join("SX", script, dr, spk, "synth_utt", f"{bname}-{script}_vocal_tract_length_change_{alpha}.wav"),
                        scaled,
                        fs,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)