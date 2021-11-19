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
- Neural oscillator

Infrastructure
- Sensor and data input/output
- Plotting

## Example

```python
from lava.lib.dnf.population.process import Population
from lava.lib.dnf.kernels.kernels import SelectiveKernel
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Convolution

# create population of 20x20 spiking neurons
dnf = Population(shape=(20, 20))

# create a selective kernel
kernel = SelectiveKernel(amp_exc=18, width_exc=[4, 4], global_inh=-15)

# apply the kernel to the population to create a DNF with a selective regime
connect(dnf.s_out, dnf.a_in, [Convolution(kernel)])
```



