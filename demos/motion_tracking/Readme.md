# Dynamic Neural Fields - Demos

This readme assumes that you have installed lava, lava-loihi and lava-dnf in the same virtual environment.

### Running the demos
The demo will run in your browser via port-forwarding. Choose a random port_num between 10000 and 20000.
(This is to avoid that multiple users try to use the same port)

#### Connect to external vlab with port-forwarding
```bash
ssh <my-vm>.research.intel-research.net -L 127.0.0.1:<port_num>:127.0.0.1:<port_num>
```

#### Activate your virtual environment
- depends on where your venv is located, e.g.:
```bash
source .venv/bin/activate
```

#### Navigate to the motion_tracking demo:
```bash
cd lava-dnf/demos/motion_tracking
```
#### Start the bokeh app
```bash
SLURM=1 LOIHI_GEN=N3C1 PARTITION=kp bokeh serve app.py --port <port_num>
```

open your browser and type:
http://localhost:<port_num>/app

As the network is pre-compiled, the demo will appear immediately, and you just need to click the "run" button to start the demo.
It is currently not possible to interrupt the demo while it is running. Please wait for the demo to terminate and click
the "close" button for all processes to terminate gracefully. 
