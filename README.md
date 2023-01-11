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
DNFs. In particular, it provides functions that generate connectivity patterns
common to DNF architectures.

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

Running DNF architectures
- On CPU (Python simulation)
- On Loihi 2

Examples demonstrating basic DNF regimes and instabilities
- Detection of input
- Selection of input
- Working memory of input
- Relational networks


## Installation
We highly recommend cloning the repository and using poetry to set up lava-dnf, provided you only want to run lava-dnf 
in simulation. This will automatically also install lava. 

Note: For INRC members who want to run lava-dnf on Loihi 2 hardware, we recommend following the 
[install instructions for the Lava-on-Loihi extension](https://intel-ncl.atlassian.net/wiki/spaces/NAP/pages/1785856001/Setup+Lava+extension+for+Loihi).

#### Linux/MacOS
```bash
$ cd $HOME
$ git clone git@github.com:lava-nc/lava-dnf.git
$ cd lava-dnf
$ curl -sSL https://install.python-poetry.org | python3
$ poetry config virtualenvs.in-project true
$ poetry install
$ source .venv/bin/activate
$ pytest
```
#### Windows (PowerShell)
```cmdlet
# Commands using PowerShell
cd $HOME
git clone https://github.com/lava-nc/lava-dnf.git
cd lava-dnf
python -m venv .venv
set-executionpolicy remotesigned  # only required on first execution
.venv\Scripts\activate
curl.exe -sSL https://install.python-poetry.org | python -
# Include the directory where poetry was installed into the PATH variable.
pip install -U pip
poetry config virtualenvs.in-project true
poetry install
pytest
```
You should expect the following output after running the unit tests with `pytest`.
```
$ pytest
================================================= test session starts =================================================
platform win32 -- Python 3.9.12, pytest-7.2.0, pluggy-1.0.0
rootdir: C:\Users\username\lava-dnf, configfile: pyproject.toml, testpaths: tests
plugins: cov-3.0.0
collected 210 items

tests\lava\lib\dnf\acceptance\test_connecting_with_ops.py .......                                                [  3%]
tests\lava\lib\dnf\acceptance\test_gauss_spike_generator.py .                                                    [  3%]
tests\lava\lib\dnf\connect\test_connect.py .............                                                         [ 10%]
tests\lava\lib\dnf\connect\test_exceptions.py .                                                                  [ 10%]
[...]
tests\lava\lib\dnf\utils\test_plotting.py ....                                                                   [ 96%]
tests\lava\lib\dnf\utils\test_validation.py .....                                                                [ 98%]
tests\lava\tutorials\test_tutorials.py sss                                                                       [100%]

=============================== warnings summary ===============================
[...]

---------- coverage: platform win32, python 3.9.12-final-0 -----------
Name                                                     Stmts   Miss  Cover   Missing
--------------------------------------------------------------------------------------
src\lava\lib\dnf\connect\connect.py                         49      0   100%
src\lava\lib\dnf\connect\exceptions.py                       6      1    83%   20
[...]
src\lava\lib\dnf\utils\plotting.py                          29      0   100%
src\lava\lib\dnf\utils\validation.py                        18      2    89%   28, 38
--------------------------------------------------------------------------------------
TOTAL                                                      806     13    98%

Required test coverage of 65.0% reached. Total coverage: 98.39%
===================================== 207 passed, 3 skipped, 2 warnings in 28.78s =====================================
```

## Example

```python
from lava.proc.lif.process import LIF
from lava.lib.dnf.kernels.kernels import SelectiveKernel
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Convolution

# Create a population of 20x20 spiking neurons.
dnf = LIF(shape=(20, 20))

# Create a selective kernel.
kernel = SelectiveKernel(amp_exc=18, width_exc=[4, 4], global_inh=-15)

# Apply the kernel to the population to create a DNF with a selective regime.
connect(dnf.s_out, dnf.a_in, [Convolution(kernel)])
```
