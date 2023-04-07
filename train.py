import os
import argparse
import numpy as np
from tqdm import tqdm
import pypianoroll
from pypianoroll import Multitrack

import torch
from torch import nn
from torch.utils.data import DataLoader

from gan.generator import Generator
from gan.discriminator import Discriminator
from gan.utils import initialize_weights
from dataset.dataset import MuseGANDataset
from trainer import train_one_step

def data_process(root_dir, beat_resolution=4, lowest_pitch=24, n_pitches=72, n_measures=4, n_samples_per_song=8):
    data = []
    song_list = os.listdir(root_dir)
    for idx in tqdm(range(len(song_list))):
        multitrack = pypianoroll.load(os.path.join(root_dir,song_list[idx]))
        for track in multitrack.tracks:
            track_name = track.name.split(' ')
            track_name = [name.lower() for name in track_name]
            if 'drums' in track_name or 'bass' in track_name or 'guitar' in track_name or 'piano' in track_name:
                continue
            multitrack.tracks.remove(track)
        if len(multitrack.tracks) < 4:
            continue
        multitrack.tracks = multitrack.tracks[0:4]
        multitrack.binarize()
        multitrack.set_resolution(beat_resolution)
        pianoroll= (multitrack.stack() > 0)
        pianoroll = pianoroll[:, :, lowest_pitch: lowest_pitch + n_pitches]
        n_total_measures = multitrack.get_max_length() // (4 * n_measures)
        candidate = n_total_measures - n_measures
        target_n_samples = min(n_total_measures // n_measures, n_samples_per_song)
        if target_n_samples > 0:
            for i in np.random.choice(candidate, target_n_samples, False):
                start = i * 4 * beat_resolution
                end = (i + n_measures) * 4 * beat_resolution
                if (pianoroll.sum(axis=(1, 2)) < 10).any():
                    continue
        else:
            continue
        data.append(torch.as_tensor(pianoroll[:, start:end], dtype=torch.float32))
    return data

def train(args):
    track_programs = [0, 0, 25, 33]
    track_names = ['Drums', 'Piano', 'Guitar', 'Bass']
    is_drums = [True, False, False, False]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    process_bar = tqdm(total=args.steps)
    file_check_points = open(args.file_checkpoints)
    tempo_array = np.full((4 * 4 * 4 * args.beat_resolution, 1), args.tempo)
    best_d_loss = 1000.0
    best_g_loss = 1000.0

    # Create dataset and data_loader
    muse_data = data_process(
        root_dir=args.data_root_dir,
        n_measures=args.n_measures,
        beat_resolution=args.beat_resolution,
        n_samples_per_song=args.n_samples_per_song,
        lowest_pitch=args.lowest_pitch,
        n_pitches=args.n_pitches
        )
    dataset = MuseGANDataset(muse_data)
    data_loader = DataLoader(dataset, args.batch_size, drop_last=True, shuffle=True)

    # Create generator and discriminator network
    generator = Generator().to(device)
    generator = generator.apply(initialize_weights)
    discriminator = Discriminator().to(device)
    discriminator = discriminator.apply(initialize_weights)

    # Define optimizer for generator and discriminator
    g_optimizer = torch.optim.Adam(
        params=generator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.9)
    )
    d_optimizer = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.9)
    )
    # Start training
    for step in range(args.steps):
        for real_samples in data_loader:
            generator.train()
            d_loss, g_loss = train_one_step(
                generator=generator,
                discriminator=discriminator,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                real_samples=real_samples,
                latent_dim=args.latent_dim,
                batch_size=args.batch_size,
                device=device
            )
            if (abs(d_loss) < best_d_loss) and (abs(g_loss) < best_g_loss):
                best_d_loss = abs(d_loss)
                best_g_loss = abs(g_loss)
                file_check_points.write(f'Checkpoint in step {step} with generator loss: {best_g_loss} and discriminator loss: {best_d_loss}')
                generator.eval()
                with torch.no_grad():
                    torch.save(generator, 'weights/generator_v3.pt')

            if step % 500 == 0:
                process_bar.set_description_str(f'd_loss = {round(d_loss, 2)}, g_loss = {round(g_loss, 2)}')
                generator.eval()
                sample_latent = torch.rand(args.batch_size, args.latent_dim)
                with torch.no_grad():
                    samples = generator(sample_latent).cpu().detach().numpy()
                samples = samples.transpose(1, 0, 2, 3).reshape(args.n_tracks, -1, args.n_pitches)
                tracks = []
                for idx, (program, is_drum, track_name) in enumerate(zip(track_programs, is_drums, track_names)):
                    pianoroll = np.pad(samples[idx] > 0.5, ((0, 0), (args.lowest_pitch, 128 - args.lowest_pitch - args.n_pitches)))
                    tracks.append(
                        pypianoroll.BinaryTrack(
                            name=track_name,
                            program=program,
                            is_drum=is_drum,
                            pianoroll=pianoroll
                        )
                    )
                m = Multitrack(
                    tracks=tracks,
                    tempo=tempo_array,
                    resolution=args.beat_resolution
                )
                m.write(f'results/step_{step}_example.mid')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='top', description='Train MuseGAN')
    parser.add_argument('--data_root_dir', type=str, default='data/cleaned_data', help='Path to Dataset')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps to train')
    parser.add_argument('--file_checkpoints', type=str, default='results/checkpoint.txt', help='Path to save checkpoint files')
    parser.add_argument('--n_measures', type=int, default=4, help='Number of measures')
    parser.add_argument('--beat_resolution', type=int, default=4, help='Number of beat resolution')
    parser.add_argument('--n_samples_per_song', type=int, default=8, help='Number of samples per song')
    parser.add_argument('--lowest_pitch', type=int, default=24, help='Value of lowest pitch')
    parser.add_argument('--n_pitches', type=int, default=72, help='Number of pitch')
    parser.add_argument('--lr', type=float, default='0.001', help='Value of learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and generated the latent vector')
    parser.add_argument('--latent_dim', type=int, default=128, help='dim of latent vector')
    parser.add_argument('--tempo', type=int, default=100, help='Value of tempo')

    opt = parser.parse_args()

    train(opt)
    
