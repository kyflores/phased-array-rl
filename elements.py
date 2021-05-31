#!/usr/bin/python
import numpy as np

class Wave:
    def __init__(self, freq_hz, amp_v, phase_rad, origin):
        self.freq_hz = freq_hz
        self.amp_v = amp_v
        self.phase_rad = phase_rad
        self.origin = origin
    
    def make_timeseries(self, ts, amp_gain = 1, phase_shift = 0):
        phase = self.phase_rad + phase_shift
        amp = self.amp_v * amp_gain
        ys = amp * np.cos(2 * np.pi * self.freq_hz * ts + phase)
        return ys
    
    def get_freq(self):
        return self.freq_hz
    
    def get_amp(self):
        return self.amp_v

    def get_phase(self):
        return self.phase_rad
    
    def get_origin(self):
        return self.origin

class Antenna:
    '''
    location_2d is an (x,y) coordinate in 2D space
    phase_init is an arbitrary static phase encapsulating unknown shifts
        PCB path length, chip variation etc.
    phase_fn is a function fn(phase_rad)->gain_scale that expresses the
        phase dependent nature of the antenna element gain.
    '''
    def __init__(self, location_2d, phase_init, gain_fn):
        self.location = location_2d
        self.phase_init = phase_init
        self.gain_fn = gain_fn
        self.phase_shift = 0
    
    def set_phase_shift(self, phase_shift):
        self.phase_shift = phase_shift

    def get_phase_shift(self):
        return self.phase_shift
    
    '''
    Calculates a time series that is the waveform out of this antenna.
    This should account for the phase shift due to distance from emitter,
        as well as the phase shift and gain loss from the antenna config.
    '''
    def get_wave_series(self, ts, wave):
        # The speed of light
        C = 299792458
        dist = np.linalg.norm(self.location - wave.get_origin())
        wvln = C / wave.get_freq()
        loc_shift = ((dist % wvln) * 2 * np.pi) / wvln

        gain = wave.get_amp() * self.gain_fn(self.phase_shift)
        phase = self.phase_init + self.phase_shift + loc_shift

        ys = wave.make_timeseries(ts, gain, phase)
        return ys

class PhasedArray:
    def __init__(self, emitters, antennas):
        self.emitters = emitters
        self.antennas = antennas
    
    def get_time_series(self, ts):
        yss = []
        res = np.zeros_like(ts)
        for em in self.emitters:
            for an in self.antennas:
                ys = an.get_wave_series(ts, em)
                yss.append(ys)
                res += ys
        
        return res, yss
    
    def get_amplitude(self, ts):
        ys, _ = self.get_time_series(ts)
        res = np.max(ys) - np.min(ys)
        return res

    def get_phases(self):
        ret = []
        for an in self.antennas:
            ret.append(an.get_phase_shift())
        
        return ret

    def set_phases(self, new_phases):
        assert(len(new_phases) == len(self.antennas))
        for x in zip(self.antennas, new_phases):
            (an, ph) = x
            an.set_phase_shift(ph)
    
