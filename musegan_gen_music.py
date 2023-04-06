import torch
import numpy as np
import yaml
from pypianoroll import BinaryTrack, Multitrack

# from midi2audio import FluidSynth
from datetime import datetime
import time

from gan.generator import *

def musegan_gen(tempo):
    with open("config.yaml") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    measures_resolution = 4 * data["beat_resolution"]
    tempo_array = np.full((4 * 4 * measures_resolution, 1), tempo)
    model = Generator()
    model = torch.load("./weights/generator_v6.pt", map_location="cpu")
    model.eval()
    with torch.no_grad():
        generated = model(torch.rand(16, 128)).detach().numpy()
        generated = generated.transpose(1, 0, 2, 3).reshape(data["n_tracks"], -1, data["n_pitches"])
    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(zip(data["programs"], data["is_drums"], data["track_names"])):
        pianoroll = np.pad(
            generated[idx] > 0.5,
            ((0, 0), (data["lowest_pitch"], 128 - data["lowest_pitch"] - data["n_pitches"]))
        )
        tracks.append(
            BinaryTrack(
                name=track_name,
                program=program,
                is_drum=is_drum,
                pianoroll=pianoroll
            )
        )
    multitrack = Multitrack(tracks=tracks, tempo=tempo_array, resolution=data["beat_resolution"])
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M%S")
    multitrack.write(f'results/results_{date_time}.mid')
    
# def wave_from_midi(midi_path: str, wave_path: str):
#     font_path =  "/fonts/font.sf2"
#     fluidsynth = FluidSynth(font_path, 16000)
#     fluidsynth.midi_to_audio(midi_path, wave_path)

if __name__ == "__main__":
    start = time.time()
    musegan_gen(70)
    end = time.time()
    print(f'Music generation done in {round((end - start), 2)}s')