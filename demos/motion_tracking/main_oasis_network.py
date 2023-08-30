from pathlib import Path
import numpy as np
import cv2
import base64
from multiprocessing import Pipe
from process_out.process import ProcessOut, DataRelayerPM
from oasis_network import MotionNetwork

from dash import Dash, html, dcc, Output, Input, no_update


def cv2_img_to_b64(img: np.ndarray) -> str:
    _, im_arr = cv2.imencode('.jpg',
                             img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)

    return im_b64.decode('utf-8')


def extract_dnf_event(dnf_event: np.ndarray) -> np.ndarray:
    return dnf_event * 254


# Lava
event_dnf_config = {
    "in_conn": {
        "weight": 8
    },
    "lif": {"du": 2000,
            "dv": 2000,
            "vth": 30

            },
    "rec_conn": {
        "amp_exc": 14,
        "width_exc": [5, 5],
        "amp_inh": -10,
        "width_inh": [10, 10]
    },
    "out_conn": {
        "weight": 20,
    }
}

selective_dnf_config = {
    "lif": {"du": 809,
            "dv": 2047,
            "vth": 30
            },
    "rec_conn": {
        "amp_exc": 7,
        "width_exc": [7, 7],
        "global_inh": -5
    }
}

dvs_file_input_config = {
    "true_height": 180,
    "true_width": 240,
    "file_path": "dvs_recording.aedat4",
    "flatten": False,
    "down_sample_factor": 8,
    "down_sample_mode": "max_pooling"
}

use_hardware = True
run_mode_normal = False

recv_pipe, send_pipe = Pipe()


oasis_network = MotionNetwork(dvs_file_input_config=dvs_file_input_config,
                              event_dnf_config=event_dnf_config,
                              selective_dnf_config=selective_dnf_config,
                              use_hardware=use_hardware,
                              run_mode_normal=run_mode_normal,
                              send_pipe=send_pipe)

# Dash
DT_MAIN_LOOP = 1000
init_img_path = "assets/wrong.png"
img_b64_prefix = 'data:image/jpg;base64,'
img_up_sample_factor = 7
img_display_width = \
    dvs_file_input_config["true_width"] * img_up_sample_factor // \
    dvs_file_input_config["down_sample_factor"]
img_display_height = \
    dvs_file_input_config["true_height"] * img_up_sample_factor // \
    dvs_file_input_config["down_sample_factor"]
num_clicks_compile = 0
num_clicks_run = 0
num_clicks_pause = 0
num_clicks_stop = 0

app = Dash(__name__)

img_original = html.Img(src=init_img_path, id="img_original",
                        width=img_display_width, height=img_display_height, className="image")

img_dnf_event_multipeak = html.Img(src=init_img_path, id="img_dnf_event_multipeak",
                                   width=img_display_width, height=img_display_height, className="image")
img_dnf_event_selective = html.Img(src=init_img_path, id="img_dnf_event_selective",
                                   width=img_display_width, height=img_display_height, className="image")

div_imgs = html.Div([
    img_original, img_dnf_event_multipeak, img_dnf_event_selective
], id="div_imgs")

button_compile = html.Button("Compile", id="button_compile", n_clicks=0,
                             disabled=run_mode_normal)
button_run = html.Button("Run", id="button_run", n_clicks=0,
                         disabled=not run_mode_normal)
button_pause = html.Button("Pause", id="button_pause", disabled=True,
                           n_clicks=0)
button_stop = html.Button("Stop", id="button_stop", disabled=True, n_clicks=0)
div_buttons = html.Div([
    button_compile, button_run, button_pause, button_stop
], id="div_buttons")

div_misc = html.Div([
    dcc.Interval(id="update_interval", interval=DT_MAIN_LOOP)
], id="div_misc")

div_all = html.Div([
    div_buttons, div_imgs,
    div_misc
], id="div_all")

app.layout = div_all


@app.callback(
    Output('button_compile', 'disabled'),
    Output('button_run', 'disabled'),
    Output('button_pause', 'disabled'),
    Output('button_stop', 'disabled'),
    Input('button_compile', 'n_clicks'),
    Input('button_run', 'n_clicks'),
    Input('button_pause', 'n_clicks'),
    Input('button_stop', 'n_clicks'),
)
def control_lava(n_clicks_compile, n_clicks_run, n_clicks_pause, n_clicks_stop):
    global num_clicks_compile, num_clicks_run, num_clicks_pause, num_clicks_stop

    print("n_clicks", n_clicks_compile, n_clicks_run, n_clicks_pause,
          n_clicks_stop)
    print("num_clicks", num_clicks_compile, num_clicks_run, num_clicks_pause,
          num_clicks_stop)

    if num_clicks_compile != n_clicks_compile:
        oasis_network.compile_network()

        num_clicks_compile = n_clicks_compile

        return True, False, True, True

    if num_clicks_run != n_clicks_run:
        oasis_network.run_network()

        num_clicks_run = n_clicks_run

        return True, True, False, False

    if num_clicks_pause != n_clicks_pause:
        oasis_network.pause_network()

        num_clicks_pause = n_clicks_pause

        return True, False, True, False

    if num_clicks_stop != n_clicks_stop:
        oasis_network.stop_network()

        num_clicks_stop = n_clicks_stop

        return True, True, True, True

    if n_clicks_run == 0:
        return run_mode_normal, not run_mode_normal, True, True
    else:
        return False, False, False, False


@app.callback(
    Output("img_original", "src"),
    Output("img_dnf_event_multipeak", "src"),
    Output("img_dnf_event_selective", "src"),
    Input('update_interval', 'n_intervals'),
)
def main_loop(update_interval):
    img_original_src = no_update
    img_dnf_event_multipeak_src = no_update
    img_dnf_event_selective_src = no_update

    if oasis_network.runtime is not None:
        if oasis_network.runtime._is_running:
            data = recv_pipe.recv()
            event_original = data["dvs_frame_ds_image"]
            dnf_multipeak = data["dnf_multipeak_rates_ds_image"]
            dnf_selective = data["dnf_selective_rates_ds_image"]

            print("we are here")
            print(event_original.shape)
            print(dnf_multipeak.shape)
            print(dnf_selective.shape)


            #event_original = extract_dnf_event(oasis_network.extractor_original.receive()).transpose()
            #dnf_multipeak = extract_dnf_event(oasis_network.extractor_dnf_multipeak.receive()).transpose()
            #dnf_selective = extract_dnf_event(oasis_network.extractor_dnf_selective.receive()).transpose()

            if event_original.sum() != 0:
                print("we enter")
                img_original_src = img_b64_prefix + cv2_img_to_b64(
                    event_original)
            if dnf_multipeak.sum() != 0:
                print("if 2")
                img_dnf_event_multipeak_src = img_b64_prefix + cv2_img_to_b64(
                    dnf_multipeak)

            if dnf_selective.sum() != 0:
                print("if 3")
                img_dnf_event_selective_src = img_b64_prefix + cv2_img_to_b64(
                    dnf_selective)

    result = (img_original_src, img_dnf_event_multipeak_src, img_dnf_event_selective_src)

    return result


if __name__ == "__main__":
    try:
        app.run_server(debug=True, port=8054)
    except:
        if oasis_network.runtime is not None:
            oasis_network.stop_network()
