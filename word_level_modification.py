import os
import soundfile as sf
import pyworld as pw
import numpy as np
import math
import scipy.interpolate
import argparse
import yaml
from tqdm import tqdm
import spacy

def read_wrd(file):
    with open(file) as f:
        lines = f.readlines()
        ali_wrd = [l.strip().split() for l in lines]
        ali_wrd = [[int(l[0]), int(l[1]), l[2]] for l in ali_wrd]
    return ali_wrd

def main(config):
    frame_period = config["frame_period"]
    pitch_shift = config["pitch_shift"]
    intensity_shift = config["intensity_shift"]


    for script in tqdm(os.listdir("SX")):
        for dr in os.listdir(os.path.join("SX", script)):
            for spk in os.listdir(os.path.join("SX", script, dr)):
                if os.path.exists(os.path.join("SX", script, dr, spk, "synth_wrd")):
                    os.system(f"rm -r {os.path.join('SX', script, dr, spk, 'synth_wrd')}")
                os.makedirs(os.path.join("SX", script, dr, spk, "synth_wrd"), exist_ok=True)

                bname = f"{dr}_{spk}"
                os.system(f"cp {os.path.join('SX', script, dr, spk, f'{script}.wav')} {os.path.join('SX', script, dr, spk, 'synth_wrd', f'{bname}-{script}.wav')}")

                orig_wav = os.path.join("SX", script, dr, spk, f"{script}.wav")
                x, fs = sf.read(orig_wav)
                orig_wrd = os.path.join("SX", script, dr, spk, f"{script}.wrd")
                ali_wrd = read_wrd(orig_wrd)

                f0, sp, ap = pw.wav2world(x, fs, frame_period=frame_period)
                
                resyn = pw.synthesize(f0, sp, ap, fs, frame_period=frame_period)
                sf.write(os.path.join("SX", script, dr, spk, "synth_wrd", f"{bname}-{script}_resyn.wav"), resyn, fs)
                # pitch shift
                shifted = pw.synthesize(
                    f0 * pitch_shift, sp, ap, fs, frame_period=frame_period
                )
                sf.write(
                    os.path.join("SX", script, dr, spk, "synth_wrd", f"{bname}-{script}_pitch_shift.wav"),
                    shifted,
                    fs,
                )

                # intensity shift
                shifted = pw.synthesize(
                    f0, sp * intensity_shift, ap, fs, frame_period=frame_period
                )
                sf.write(
                    os.path.join("SX", script, dr, spk, "synth_wrd", f"{bname}-{script}_intensity_shift.wav"),
                    shifted,
                    fs,
                )

                for i, w in enumerate(ali_wrd):
                    start_frame = int(math.floor(w[0]) / (fs * frame_period / 1000))
                    end_frame = int(math.ceil(w[1]) / (fs * frame_period / 1000))
                    # word level pitch shift
                    f0_shift = f0.copy()
                    f0_shift[start_frame:end_frame] = f0[start_frame:end_frame] * pitch_shift
                    word_pitch_shift = pw.synthesize(
                        f0_shift, sp, ap, fs, frame_period=frame_period
                    )
                    sf.write(
                        os.path.join("SX", script, dr, spk, "synth_wrd", f"{bname}-{script}_word_pitch_shift_{i}_{w[2]}.wav"),
                        word_pitch_shift,
                        fs,
                    )
                    # word level intensity shift
                    sp_shift = sp.copy()
                    sp_shift[start_frame:end_frame] = sp[start_frame:end_frame] * intensity_shift
                    word_intensity_shift = pw.synthesize(
                        f0, sp_shift, ap, fs, frame_period=frame_period
                    )
                    sf.write(
                        os.path.join("SX", script, dr, spk, "synth_wrd", f"{bname}-{script}_word_intensity_shift_{i}_{w[2]}.wav"),
                        word_intensity_shift,
                        fs,
                    )

                




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)