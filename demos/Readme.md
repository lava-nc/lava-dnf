# Demos
In order to run a demo, you'll need to follow these steps: 
#### 1. Connect to vLab 
#### 2. Execute the following commands: 
```bash
$ port_num=$(python <<< 'import random; print(random.randint(5006, 5106))')
$ echo $port_num
Copy & use to replace <port_num> when running the Bokeh server at the end 
$ ssh <my-vm>.research.intel-research.net -L 127.0.0.1:$port_num:127.0.0.1:$port_num 
```
#### 3. Activate your virtual environment that has lava, lava-loihi and lava-dnf installed 
#### 4. Install bokeh and scipy
```bash
$ pip install scipy
$ pip intall bokeh
```
#### 4. Navigate to the Demo you want to try
```bash
$  cd lava-dnf/demos/<demo-you-want-to-try>  
```
#### 5. Run the Bokeh server by executing the following command
```bash
$  SLURM=1 LOIHI_GEN=N3B3 PARTITION=oheogulch bokeh serve app.py --port <port_num> 
```

#### 7. On your local machine, open your browser, and open the following url:
http://localhost:<port_num>/app 

You should see the demo component static. 

7. Enjoy!