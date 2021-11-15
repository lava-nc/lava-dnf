from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.compiler.compiler import Compiler
from lava.magma.runtime.runtime import Runtime
from lava.magma.core.process.message_interface_enum import ActorType

from lava.lib.dnf.inputs.spike_source.process import SpikeSource
from lava.lib.dnf.inputs.inputs import GaussInputPattern, SpikeInputGenerator


def main():
    num_steps = 1

    input_pattern = GaussInputPattern(shape=(30,), amplitude=1.0, mean=15, stddev=5)
    spike_generator = SpikeInputGenerator(input_pattern)

    spike_source = SpikeSource(generator=spike_generator)

    # spike_source.run(condition=RunSteps(num_steps=num_steps),
    #                  run_cfg=Loihi1SimCfg())
    # spike_source.stop()

    # create a compiler
    compiler = Compiler()

    # compile the Process (and all connected Processes) into an executable
    executable = compiler.compile(spike_source, run_cfg=Loihi1SimCfg())

    # create and initialize a runtime
    runtime = Runtime(run_cond=RunSteps(num_steps=num_steps), exe=executable,
                      message_infrastructure_type=ActorType.MultiProcessing)

    # TODO : problematic line
    runtime.initialize()

    # # start execution
    # runtime.start(run_condition=RunSteps(num_steps=num_steps))
    #
    # # stop execution
    # runtime.stop()


if __name__ == '__main__':
    main()