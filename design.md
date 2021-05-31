# Abstract model of an antenna array in 2D space
## State
- Static state:
    - Position of antenna elements
    - Random phase angle per antenna
    - Position of radiator
    - Wave from radiator
- State Transition:
    - In: the current phase shift programmed to each antenna (maybe a few time steps of history)
      and corresponding power
    - Out: New phase shift and corresponding power
    - Reward: Different of curent power from theoretical maximum power (calculated from radiators)
## Classes
- Wave:
    - A single wave in A*cos(2pi*w*t + B) form
    - Can calcuate the time series of the wave
- Antenna:
    - An antenna array element in 2D space
    - Applies phase shift
    - Calcuates a wave timeseries according to antenna characteristics like...
        - Position relative to the transmitters
        - Antenna amp gain
        - Phase dependent gain
        - Programmed phase shift
        - Innate phase shift (random phase angle)
- Radiator:
    - Represents a single point radiating a wave.
- PhasedArray:
    - Owns a collection of antennas
    - Sums up the total signal from all antennas
    - Calculates total antenna power in dBm.