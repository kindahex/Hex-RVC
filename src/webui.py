import argparse
import gc
import hashlib
import json
import os
import shlex
import subprocess
from contextlib import suppress
from urllib.parse import urlparse, parse_qs

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import sox
import yt_dlp
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment

from mdx import run_mdx
from rvc import Config, load_hubert, get_vc, rvc_infer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')
rvc_output_dir = os.path.join(BASE_DIR, 'rvc_output')


def get_youtube_video_id(url, ignore_playlist=True):
    """
    Examples:
    http://youtu.be/SA2iWivDJiE
    http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    http://www.youtube.com/embed/SA2iWivDJiE
    http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            # use case: get playlist id not current video in playlist
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/':
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]

    # returns None for invalid YouTube url
    return None


def yt_download(link):
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl='%(title)s.mp3')

    return download_path


def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)


def get_rvc_model(voice_model, is_webui):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f'No model file exists in {model_dir}.'
        raise_exception(error_msg, is_webui)

    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''


def get_audio_paths(song_dir):
    orig_song_path = None
    instrumentals_path = None
    main_vocals_dereverb_path = None
    backup_vocals_path = None

    for file in os.listdir(song_dir):
        if file.endswith('_Instrumental.wav'):
            instrumentals_path = os.path.join(song_dir, file)
            orig_song_path = instrumentals_path.replace('_Instrumental', '')

        elif file.endswith('_Vocals_Main_DeReverb.wav'):
            main_vocals_dereverb_path = os.path.join(song_dir, file)

        elif file.endswith('_Vocals_Backup.wav'):
            backup_vocals_path = os.path.join(song_dir, file)

    return orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path


def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)

    # check if mono
    if type(wave[0]) != np.ndarray:
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"')
        subprocess.run(command)
        return stereo_path
    else:
        return audio_path


def pitch_shift(audio_path, pitch_change):
    output_path = f'{os.path.splitext(audio_path)[0]}_p{pitch_change}.wav'
    if not os.path.exists(output_path):
        y, sr = sf.read(audio_path)
        tfm = sox.Transformer()
        tfm.pitch(pitch_change)
        y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
        sf.write(output_path, y_shifted, sr)

    return output_path


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:11]


def display_progress(message, percent, is_webui, progress=None):
    if is_webui:
        progress(percent, desc=message)
    else:
        print(message)


def preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress=None):
    keep_orig = False
    if input_type == 'yt':
        display_progress('[~] Downloading song...', 0, is_webui, progress)
        song_link = song_input.split('&')[0]
        orig_song_path = yt_download(song_link)
    elif input_type == 'local':
        orig_song_path = song_input
        keep_orig = True
    else:
        orig_song_path = None

    song_output_dir = os.path.join(output_dir, song_id)
    orig_song_path = convert_to_stereo(orig_song_path)

    display_progress('[~] Separating Vocals from Instrumental...', 0.1, is_webui, progress)
    vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx'), orig_song_path, denoise=True, keep_orig=keep_orig)

    display_progress('[~] Separating Main Vocals from Backup Vocals...', 0.2, is_webui, progress)
    backup_vocals_path, main_vocals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR_MDXNET_KARA_2.onnx'), vocals_path, suffix='Backup', invert_suffix='Main', denoise=True)

    display_progress('[~] Applying DeReverb to Vocals...', 0.3, is_webui, progress)
    _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), main_vocals_path, invert_suffix='DeReverb', exclude_main=True, denoise=True)

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path


def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui):
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model, is_webui)
    device = 'cuda:0'
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    # convert main vocals
    rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)
    del hubert_model, cpt
    gc.collect()


def add_audio_effects(audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping):
    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'

    # Initialize audio effects plugins
    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(ratio=4, threshold_db=-15),
            Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping)
         ]
    )

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    return output_path


def combine_audio(audio_paths, output_path, main_gain, backup_gain, inst_gain, output_format):
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    backup_vocal_audio = AudioSegment.from_wav(audio_paths[1]) - 6 + backup_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) - 7 + inst_gain
    main_vocal_audio.overlay(backup_vocal_audio).overlay(instrumental_audio).export(output_path, format=output_format)


def song_cover_pipeline(song_input, voice_model, pitch_change, keep_files,
                        is_webui=0, main_gain=0, backup_gain=0, inst_gain=0, index_rate=0.5, filter_radius=3,
                        rms_mix_rate=0.25, f0_method='rmvpe', crepe_hop_length=128, protect=0.33, pitch_change_all=0,
                        reverb_rm_size=0.15, reverb_wet=0.2, reverb_dry=0.8, reverb_damping=0.7, output_format='mp3',
                        progress=gr.Progress()):
    try:
        if not song_input or not voice_model:
            raise_exception('Ensure that the song input field and voice model field is filled.', is_webui)

        display_progress('[~] Starting AI Cover Generation Pipeline...', 0, is_webui, progress)

        with open(os.path.join(mdxnet_models_dir, 'model_data.json')) as infile:
            mdx_model_params = json.load(infile)

        # if youtube url
        if urlparse(song_input).scheme == 'https':
            input_type = 'yt'
            song_id = get_youtube_video_id(song_input)
            if song_id is None:
                error_msg = 'Invalid YouTube url.'
                raise_exception(error_msg, is_webui)

        # local audio file
        else:
            input_type = 'local'
            song_input = song_input.strip('\"')
            if os.path.exists(song_input):
                song_id = get_hash(song_input)
            else:
                error_msg = f'{song_input} does not exist.'
                song_id = None
                raise_exception(error_msg, is_webui)

        song_dir = os.path.join(output_dir, song_id)

        if not os.path.exists(song_dir):
            os.makedirs(song_dir)
            orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress)

        else:
            vocals_path, main_vocals_path = None, None
            paths = get_audio_paths(song_dir)

            # if any of the audio files aren't available or keep intermediate files, rerun preprocess
            if any(path is None for path in paths) or keep_files:
                orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress)
            else:
                orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path = paths

        # pitch_change = pitch_change + pitch_change_all
        ai_vocals_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]}.wav')
        ai_cover_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]} ({voice_model} Ver).{output_format}')

        if not os.path.exists(ai_vocals_path):
            display_progress('[~] Converting voice using RVC...', 0.5, is_webui, progress)
            voice_change(voice_model, main_vocals_dereverb_path, ai_vocals_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui)

        display_progress('[~] Applying audio effects to Vocals...', 0.8, is_webui, progress)
        ai_vocals_mixed_path = add_audio_effects(ai_vocals_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping)

        if pitch_change_all != 0:
            display_progress('[~] Applying overall pitch change', 0.85, is_webui, progress)
            instrumentals_path = pitch_shift(instrumentals_path, pitch_change_all)
            backup_vocals_path = pitch_shift(backup_vocals_path, pitch_change_all)

        display_progress('[~] Combining AI Vocals and Instrumentals...', 0.9, is_webui, progress)
        combine_audio([ai_vocals_mixed_path, backup_vocals_path, instrumentals_path], ai_cover_path, main_gain, backup_gain, inst_gain, output_format)

        if not keep_files:
            display_progress('[~] Removing intermediate audio files...', 0.95, is_webui, progress)
            intermediate_files = [vocals_path, main_vocals_path, ai_vocals_mixed_path]
            if pitch_change_all != 0:
                intermediate_files += [instrumentals_path, backup_vocals_path]
            for file in intermediate_files:
                if file and os.path.exists(file):
                    os.remove(file)

        return ai_cover_path

    except Exception as e:
        raise_exception(str(e), is_webui)



    



def get_rvc_models():
    """
    Detects folders model inside the rvc_models_dir and returns a list.
    """
    if os.path.exists(rvc_models_dir) and os.path.isdir(rvc_models_dir):
        folders = [f for f in os.listdir(rvc_models_dir) if os.path.isdir(os.path.join(rvc_models_dir, f))]
        return folders if folders else ["No folders found"]
    else:
        return ["Directory does not exist"]


def update_dropdown():
        return gr.update(choices=get_rvc_models())
    


# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Audio Conversion Tool")
    rvc_dirname = gr.Dropdown(choices=[], label="Select Model", interactive=True)        
    refresh_button = gr.Button("Refresh")
    with gr.Tab("Inference"):
        with gr.Row():
            song_input = gr.Textbox(label="Audio Input (URL or Path)", value="https://youtu.be/15Cq6FVqWMI?si=frcPY_zSptEWRaBg")
            pitch_algo = gr.Dropdown(label="Pitch Detection Algorithm", 
                                     choices=["mangio-crepe", "rmvpe", "fcpe", "rmvpe+", "hybrid[fcpe+rmvpe]"], 
                                     value="hybrid[fcpe+rmvpe]")
        with gr.Row():
          pitch_change = gr.Slider(label="Pitch Change Vocal", value=0, minimum=-12, maximum=12, step=0.1)
          pitch_change_all = gr.Slider(label="Pitch Change All", value=0, minimum=-12, maximum=12, step=0.1)
          with gr.Row():
            index_rate = gr.Slider(label="Index Rate", value=0.5, minimum=0, maximum=1, step=0.01)
            filter_radius = gr.Number(label="Filter Radius", value=3, precision=0)
            crepe_hop_length = gr.Number(label="Crepe Hop Length", value=128, precision=0)
            protect = gr.Slider(label="Protect", value=0.33, minimum=0, maximum=1, step=0.01)
            remix_mix_rate = gr.Slider(label="Remix Mix Rate", value=0.25, minimum=0, maximum=1, step=0.01)
        
        with gr.Row():
            main_vol = gr.Number(label="Main Volume", value=0, precision=0)
            backup_vol = gr.Number(label="Backup Volume", value=0, precision=0)
            inst_vol = gr.Number(label="Instrumental Volume", value=0, precision=0)
            
        with gr.Row():
            reverb_size = gr.Slider(label="Reverb Size", value=0.15, minimum=0, maximum=1, step=0.01)
            reverb_wetness = gr.Slider(label="Reverb Wetness", value=0.2, minimum=0, maximum=1, step=0.01)
            reverb_dryness = gr.Slider(label="Reverb Dryness", value=0.8, minimum=0, maximum=1, step=0.01)
            reverb_damping = gr.Slider(label="Reverb Damping", value=0.7, minimum=0, maximum=1, step=0.01)
        output_format = gr.Radio(label="Output Format", choices=["mp3", "wav"], value="mp3")
    
    generate_button = gr.Button("Generate")
    audio_output = gr.Audio(label="Generated Audio")

    
    refresh_button.click(fn=update_dropdown, outputs=rvc_dirname)
    generate_button.click(
        song_cover_pipeline, 
        inputs=[song_input, rvc_dirname, pitch_algo, pitch_change, pitch_change_all, index_rate, filter_radius, 
                crepe_hop_length, protect, remix_mix_rate, main_vol, backup_vol, inst_vol, reverb_size, 
                reverb_wetness, reverb_dryness, reverb_damping, output_format], 
        outputs=[audio_output]
    )

# Launch the app
app.launch(share=True)
