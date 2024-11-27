import gradio as gr
import subprocess
import os
from src.main import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')



# Define the function to execute the process
def generate_audio(song_input, rvc_dirname, pitch_algo, pitch_change, pitch_change_all, index_rate, filter_radius, 
                   crepe_hop_length, protect, remix_mix_rate, main_vol, backup_vol, inst_vol, reverb_size, 
                   reverb_wetness, reverb_dryness, reverb_damping, output_format):
    output_file = f"output.{output_format}"
    command = [
        "python",
        "src/main.py",
        "-i", song_input,
        "-dir", rvc_dirname,
        "-p", str(pitch_change),
        "-k",
        "-ir", str(index_rate),
        "-fr", str(filter_radius),
        "-rms", str(remix_mix_rate),
        "-palgo", pitch_algo,
        "-hop", str(crepe_hop_length),
        "-pro", str(protect),
        "-mv", str(main_vol),
        "-bv", str(backup_vol),
        "-iv", str(inst_vol),
        "-pall", str(pitch_change_all),
        "-rsize", str(reverb_size),
        "-rwet", str(reverb_wetness),
        "-rdry", str(reverb_dryness),
        "-rdamp", str(reverb_damping),
        "-oformat", output_format,
        "-o", output_file
    ]

    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    _, stderr = process.communicate()
    process.wait()
    
    # Check if the file was created
    if os.path.exists(ai_cover_path):
        return ai_cover_path
    else:
        return f"Error: {stderr}"




def get_rvc_models():
    """
    Detects folders model inside the rvc_models_dir and returns a list.
    """
    if os.path.exists(rvc_models_dir) and os.path.isdir(rvc_models_dir):
        folders = [f for f in os.listdir(rvc_models_dir) if os.path.isdir(os.path.join(rvc_models_dir, f))]
        return folders if folders else ["No folders found"]
    else:
        return ["Directory does not exist"]



# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Audio Conversion Tool")
    rvc_dirname = gr.Dropdown(choices=[], label="Select Folder", interactive=True)        
    refresh_button = gr.Button("Refresh")
    with gr.Tab("Inference"):
        with gr.Row():
            song_input = gr.Textbox(label="Audio Input (URL or Path)", value="https://youtu.be/15Cq6FVqWMI?si=frcPY_zSptEWRaBg")
            pitch_algo = gr.Dropdown(label="Pitch Detection Algorithm", 
                                     choices=["mangio-crepe", "rmvpe", "fcpe", "rmvpe+", "hybrid[fcpe+rmvpe]"], 
                                     value="hybrid[fcpe+rmvpe]")
        with gr.Row():
            pitch_change = gr.Number(label="Pitch Change", value=0, precision=0)
            pitch_change_all = gr.Number(label="Pitch Change All", value=0, precision=0)
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

    generate_button.click(
        generate_audio, 
        inputs=[song_input, rvc_dirname, pitch_algo, pitch_change, pitch_change_all, index_rate, filter_radius, 
                crepe_hop_length, protect, remix_mix_rate, main_vol, backup_vol, inst_vol, reverb_size, 
                reverb_wetness, reverb_dryness, reverb_damping, output_format], 
        outputs=[audio_output]
    )

# Launch the app
app.launch()
