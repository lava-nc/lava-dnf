import numpy as np
import matplotlib.pyplot as plt
import typing as ty

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

from lava.lib.dnf.inputs.inputs import gauss


class GaussPattern(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        shape = kwargs.pop("shape")
        self._shape = Var(shape=(len(shape), ), init=np.array(shape))

        self._amplitude = Var(shape=(1,), init=kwargs.pop("amplitude"))
        self._mean = Var(shape=(len(shape),), init=kwargs.pop("mean"))
        self._stddev = Var(shape=(len(shape),), init=kwargs.pop("stddev"))

        self._pattern = Var(shape=shape, init=0)

        self._changed = Var(shape=(1, ), init=True)

        self.a_out = OutPort(shape=(2, 2))

    def _validate_mean(self, mean: ty.Union[float, ty.List[float]]):
        if isinstance(mean, float) or isinstance(mean, int):
            mean = float(mean)
            mean = [mean]

        if len(mean) == 1:
            mean_val = mean[0]
            mean = [mean_val for _ in range(self.shape.shape[0])]
        elif len(mean) > 1:
            if len(mean) != self.shape.shape[0]:
                raise ValueError("<mean> parameter has length different from shape dimensionality")
        else:
            raise ValueError("<mean> parameter cannot be empty")

        return np.array(mean)

    def _validate_stddev(self, stddev: ty.Union[float, ty.List[float]]):
        if isinstance(stddev, float) or isinstance(stddev, int):
            stddev = float(stddev)
            stddev = [stddev]

        if len(stddev) == 1:
            stddev_val = stddev[0]
            stddev = [stddev_val for _ in range(self.shape.shape[0])]
        elif len(stddev) > 1:
            if len(stddev) != self.shape.shape[0]:
                raise ValueError("<stddev> parameter has length different from shape dimensionality")
        else:
            raise ValueError("<stddev> parameter cannot be empty")

        return np.array(stddev)

    def _update(self):
        self._changed.set(np.array([True]))

    @property
    def shape(self) -> np.ndarray:
        try:
            return self._shape.get()
        except AttributeError:
            return None

    @property
    def amplitude(self) -> np.ndarray:
        try:
            return self._amplitude.get()
        except AttributeError:
            return None

    @amplitude.setter
    def amplitude(self, amplitude: float):
        self._amplitude.set(np.array([amplitude]))
        self._update()

    @property
    def mean(self) -> np.ndarray:
        try:
            return self._mean.get()
        except AttributeError:
            return None

    @mean.setter
    def mean(self, mean: ty.Union[float, ty.List[float]]):
        mean = self._validate_mean(mean)
        self._mean.set(mean)
        self._update()

    @property
    def stddev(self) -> np.ndarray:
        try:
            return self._stddev.get()
        except AttributeError:
            return None

    @stddev.setter
    def stddev(self, stddev: ty.Union[float, ty.List[float]]):
        stddev = self._validate_stddev(stddev)
        self._stddev.set(stddev)
        self._update()


@implements(proc=GaussPattern, protocol=LoihiProtocol)
@requires(CPU)
class GaussPatternProcessModel(PyLoihiProcessModel):
    _shape: np.ndarray = LavaPyType(np.ndarray, int)

    _amplitude: np.ndarray = LavaPyType(np.ndarray, float)
    _mean: np.ndarray = LavaPyType(np.ndarray, float)
    _stddev: np.ndarray = LavaPyType(np.ndarray, float)

    _pattern: np.ndarray = LavaPyType(np.ndarray, float)

    _changed: np.ndarray = LavaPyType(np.ndarray, bool)

    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        if self._changed[0]:
            self._pattern = gauss(shape=self._shape,
                                 domain=None,
                                 amplitude=self._amplitude[0],
                                 mean=self._mean,
                                 stddev=self._stddev)
            self._changed[0] = False
            self.a_out.send(np.ones((2, 2)))
            # self.a_out.send(self._pattern)


TIME_STEPS_PER_MINUTE = 6000.0


class SpikeGenerator(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        shape = kwargs.pop("shape")

        self.inter_spike_distances = Var(shape=shape, init=np.inf)
        self.last_spiked = Var(shape=shape, init=-np.inf)

        self.spikes = Var(shape=shape, init=0)

        self.a_in = InPort(shape=(2, 2))
        self.s_out = OutPort(shape=shape)


@implements(proc=SpikeGenerator, protocol=LoihiProtocol)
@requires(CPU)
class SpikeGeneratorProcessModel(PyLoihiProcessModel):
    inter_spike_distances: np.ndarray = LavaPyType(np.ndarray, float)
    last_spiked: np.ndarray = LavaPyType(np.ndarray, float)

    ts_last_changed: int = 0

    spikes: np.ndarray = LavaPyType(np.ndarray, bool)

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)

    def _compute_distances(self, pattern):
        distances = np.zeros_like(pattern)

        for idx, item in enumerate(pattern.flat):
            if item > 0.5:
                distance = np.int(np.rint(TIME_STEPS_PER_MINUTE / item))

                distances.flat[idx] = distance
            else:
                distances.flat[idx] = np.inf

        return distances

    def _generate_spikes(self,
                         time_step: int) -> np.ndarray:
        proba = np.zeros(self.spikes.shape)
        result = np.zeros(self.spikes.shape, np.bool)

        idx_zeros = self.inter_spike_distances == np.inf
        idx_proba = self.last_spiked == - np.inf

        proba[idx_proba] = float(time_step - self.ts_last_changed) / self.inter_spike_distances[idx_proba]
        proba[idx_zeros] = 0

        first_spikes = np.random.binomial(1, proba)

        distances_last_spiked = time_step - self.last_spiked

        result[idx_proba] = first_spikes[idx_proba] == 1
        result[~idx_proba] = distances_last_spiked[~idx_proba] >= self.inter_spike_distances[~idx_proba]

        self.last_spiked[result] = time_step

        return result

    def run_spk(self):
        if self.a_in.probe():
            self.ts_last_changed = self.current_ts
            self.last_spiked[:] = - np.inf
            pattern = self.a_in.recv()
            self.inter_spike_distances = self._compute_distances(pattern)

        self.spikes = self._generate_spikes(time_step=self.current_ts)
        self.s_out.send(self.spikes)


def main():
    num_steps = 250

    gauss_pattern = GaussPattern(shape=(60,), amplitude=500.0, mean=50, stddev=15)
    spike_generator = SpikeGenerator(shape=(60,))

    gauss_pattern.out_ports.a_out.connect(spike_generator.in_ports.a_in)

    spikes = np.zeros((gauss_pattern._shape.get()[0], 5*num_steps))

    for i in range(num_steps):
        print(f"Step {i}")

        spike_generator.run(condition=RunSteps(num_steps=1),
                               run_cfg=Loihi1SimCfg(select_sub_proc_model=True))

        spikes[:, i] = spike_generator.vars.spikes.get()

    gauss_pattern.mean = 40

    for i in range(num_steps):
        print(f"Step {i+num_steps}")

        spike_generator.run(condition=RunSteps(num_steps=1),
                               run_cfg=Loihi1SimCfg(select_sub_proc_model=True))

        spikes[:, i+num_steps] = spike_generator.vars.spikes.get()

    gauss_pattern.mean = 30

    for i in range(num_steps):
        print(f"Step {i+(2*num_steps)}")

        spike_generator.run(condition=RunSteps(num_steps=1),
                               run_cfg=Loihi1SimCfg(select_sub_proc_model=True))

        spikes[:, i+(2*num_steps)] = spike_generator.vars.spikes.get()

    gauss_pattern.mean = 20

    for i in range(num_steps):
        print(f"Step {i+(3*num_steps)}")

        spike_generator.run(condition=RunSteps(num_steps=1),
                               run_cfg=Loihi1SimCfg(select_sub_proc_model=True))

        spikes[:, i+(3*num_steps)] = spike_generator.vars.spikes.get()

    gauss_pattern.mean = 10

    for i in range(num_steps):
        print(f"Step {i+(4*num_steps)}")

        spike_generator.run(condition=RunSteps(num_steps=1),
                               run_cfg=Loihi1SimCfg(select_sub_proc_model=True))

        spikes[:, i+(4*num_steps)] = spike_generator.vars.spikes.get()

    spike_generator.stop()

    plt.figure(figsize=(20, 20), facecolor="grey")

    for idx, spike_array in enumerate(spikes):
        spike_positions = np.where(spike_array == 1.0)
        plt.eventplot(spike_positions, lineoffsets=idx, linelengths=0.5)

    plt.show()


if __name__ == "__main__":
    main()
