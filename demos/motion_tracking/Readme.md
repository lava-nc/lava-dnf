# Dynamic Neural Fields - Demos

This readme assumes that you have installed lava, lava-loihi and lava-dnf in the same virtual environment. Additionally, you should check if you downloaded the input recording which is stored using git lfs. Check if the file size is reasonable.

```bash
lava-dnf/demos/motion_tracking$ ls -l
-rw-r--r-- 1 <user> <group>  93353870 Sep 18 08:01 dvs_recording.aedat4
```
If the file size is only 133, then use the following command before running:
```bash
lava-dnf/demos/motion_tracking$ git lfs pull
```

### Running the demos
The demo will run in your browser via port-forwarding. Choose a random port_num between 10000 and 20000.
(This is to avoid that multiple users try to use the same port)

#### Connect to external vlab with port-forwarding
```bash
ssh <my-vm>.research.intel-research.net -L 127.0.0.1:<port_num>:127.0.0.1:<port_num>
```

#### Activate your virtual environment
Location of your virtual enviornment might differ.
```bash
source lava/lava_nx_env/bin/activate
```

#### Navigate to the motion_tracking demo:
```bash
cd lava-dnf/demos/motion_tracking
```
#### Start the bokeh app
```bash
bokeh serve main_motion_tracking.py --port <port_num>
```

open your browser and type:
http://localhost:<port_num>/main_motion_tracking

As the network is pre-compiled, the demo will appear immediately, and you just need to click the "run" button to start the demo.
It is currently not possible to interrupt the demo while it is running. Please wait for the demo to terminate and click the "close" button for all processes to terminate gracefully. 
