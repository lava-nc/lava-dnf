# Dynamic Neural Fields

## Introduction

Dynamic Neural Fields (DNF) are neural attractor networks that generate
stabilized activity patterns in recurrently connected populations of neurons.
These activity patterns form the basis of neural representations, decision
making, working memory, and learning. DNFs are the fundamental
building block of [dynamic field theory](https://dynamicfieldtheory.org),
a mathematical and conceptual framework for modeling cognitive processes in
a closed behavioral loop.

![2D DNF tracking bias input](https://user-images.githubusercontent.com/5708333/135443996-7492b968-277a-4397-9b1c-597b7af4a699.gif)<br>
*Voltage of a selective dynamic neural field tracking moving input*

## What is lava-dnf?

lava-dnf is a library within the Lava software framework. The main building 
blocks in Lava are processes. lava-dnf provides
processes and other software infrastructure to build architectures composed of
DNFs. It also provides tools to direct sensory input to
neural architectures and to read output, for instance for motor control.

The primary focus of lava-dnf today is on robotic applications: sensing and
perception, motion control, behavioral organization, map formation, and
autonomous (continual) learning. Neuromorphic hardware provides significant
gains in both processing speed and energy efficiency compared to conventional
implementations of DNFs on a CPU or GPU (e.g., using
[cedar](https://cedar.ini.rub.de) or [cosivina](https://github.com/cosivina)).

## Key features

Building DNF architectures
- Based on spiking neurons
- DNF dimensionality support for 0D, 1D, 2D, and 3D
- Recurrent connectivity based on kernel functions
- Forward connectivity to connect multiple DNFs
- Structured input from spike generators

Examples demonstrating basic DNF regimes and instabilities
- Detection of input
- Selection of input
- Working memory of input

## Installation
### Cloning lava-dnf and Running from Source
We highly recommend cloning the repository and using pybuilder to setup lava. You will need to install pybuilder for the same.

Note: We assume you have already setup Lava with virtual environment. Test your PYTHONPATH using `echo $PYTHONPATH` and
ensure 'lava/src' is the first entry that precedes any additional Lava library src paths. 

#### [Linux/MacOS]
```bash
$ git clone git@github.com:lava-nc/lava-dnf.git
$ cd lava-dnf
$ pip install -r requirements.txt
$ export PYTHONPATH=$PYTHONPATH:$(pwd)/src
$ pyb -E unit
```

You should expect the following output after running the unit tests:
```bash
PyBuilder version 0.13.4
Build started at 2022-03-01 06:36:09
------------------------------------------------------------
[INFO] Installing or updating plugin "pypi:pybuilder_bandit, module name 'pybuilder_bandit'"
[...]
[INFO] Running Twine check for generated artifacts
------------------------------------------------------------
BUILD SUCCESSFUL
------------------------------------------------------------
Build Summary
Project: lava-dnf
Version: 0.1.0
Base directory: /home/user/lava-dnf
Environments: unit
Tasks: prepare [130648 ms] compile_sources [0 ms] run_unit_tests [17550 ms] analyze [837 ms] package [115 ms] run_integration_tests [0 ms] verify [0 ms] coverage [22552 ms] publish [6177 ms]
Build finished at 2022-03-01 06:39:16
Build took 186 seconds (186983 ms)

```

## Example

```python
from lava.proc.lif.process import LIF
from lava.lib.dnf.kernels.kernels import SelectiveKernel
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Convolution

# create population of 20x20 spiking neurons
dnf = LIF(shape=(20, 20))

# create a selective kernel
kernel = SelectiveKernel(amp_exc=18, width_exc=[4, 4], global_inh=-15)

# apply the kernel to the population to create a DNF with a selective regime
connect(dnf.s_out, dnf.a_in, [Convolution(kernel)])
```
