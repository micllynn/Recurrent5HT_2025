"""
Functions related to least-squares fitting for the STP kernel.

mbfl 20.7
"""
import numpy as np
from types import SimpleNamespace

import brian2 as b2
from brian2 import NeuronGroup, Synapses, PoissonInput, PoissonGroup, \
    TimedArray, SpikeGeneratorGroup, network_operation, prefs
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor

# brian2 runtime parameters
prefs.codegen.target = 'cython'
b2.defaultclock.dt = 0.1 * b2.ms


# * Helper functions:
# ---------------------------------------------------------------


def gen_lhbin_lambda(t_max,
                     dt,
                     t_delay=100,
                     fr_base=0,
                     fr_max=50,
                     t_fr_tau=100):
    '''
    Generates an exponentially decaying lambda, representing
    inhomogeneous poisson spiketrains from LHb.

    Parameters
    ------
    t_max : float
        The maximum time to simulate until (msec)
    dt : float
        The time interval (msec)
    t_delay : float (default 100)
        The time to delay until lhb activation (msec)
    fr_base : float (default 0)
        Baseline firing rate (Hz)
    fr_max : float (default 50)
        The maximum firing rate reached (Hz)
    t_fr_tau : float (default 100)
        The tau of the firing rate decay (msec)

    Returns
    ------
    lambda_pois : np.ndarray
        A time-varying vector of lambdas (exponentially decaying) which
        represent an inhom. poisson rate
    '''
    time = np.arange(0, t_max, dt)
    exponential = (np.exp(-time / t_fr_tau)) * fr_max + fr_base

    firing_delay_inds = int(t_delay / dt)
    lambda_pois = TimedArray(
        np.pad(exponential, (int(np.ceil(firing_delay_inds)), 0),
               'constant',
               constant_values=fr_base)[0:len(exponential)] * b2.Hz,
        dt=dt * b2.ms)

    return lambda_pois


def gen_lhbin_syn_times(t_max, dt, t_onset=100, t_offset=500, fr=20):
    '''
    Generates a TimedArray of a barrage of synaptic input from lhb to drn,
    at some constant rate.

    Parameters
    --------
    t_max : float
        Maximum time to simulate until (msec)
    dt : float
        The time interval (msec)
    t_onset : float (default 100)
        Start time of LHb stim (msec)
    t_offset : float (default 500)
        End time of LHb stim (msec)
    fr : float (default 20)
        LHb input frequency (Hz)
            - Note that this is constant

    Returns
    ------
    t_stims : np.array
    Returns the times (ms) of a barrage of synaptic inputs

    '''

    time = np.arange(0, t_max, dt)

    syn_input_ = np.zeros(len(time))

    n_stims = int((t_offset - t_onset) / 1000 * fr)
    t_stims = np.array(t_onset + (np.arange(n_stims) * (1 / fr) * 1000),
                       dtype=np.int)

    ind_stims = np.array(t_stims / dt, dtype=np.int)
    syn_input_[ind_stims] = 1

    # syn_input = TimedArray(syn_input_, dt=dt * b2.ms)

    return t_stims


def gen_lhb_burstevents(t_tot=2000, t_lhbdelay=1000, t_lhbon=400,
                        n_neurs=1000, rate=20, burst_frac=0.4, dt=0.5):
    """
    Generate a list of inds and times of LHb spiking with some burst fraction.
    To be used to initialize a SpikeGeneratorGroup.
    """
    inds = np.arange(t_tot, t_lhbdelay)
    return


def make_eqs(new_syn_model=True, simplify=False,
             add_pfc=True):
    """
    Return a SimpleNamespace consisting of equation strings
    for excitatory LIF dynamics (.excit_lif_dynamics), the
    5-HT1AR synapse model (.syn_ee_model), and the
    5-HT1AR synapse instructions upon a presynaptic spike
    (.syn_ee_on_pre).

    Parameters
    ----------
    new_syn_model : bool
        Whether to use the new synapse model (linear-nonlinear kernel
        convolution from Rossbroich&Naud 2020) or the old model
        (calcium accumulation presynaptically, coupled to Pr change;
        5-HT binding with Hill equation postsynaptically).

    simplify : bool
        If new_syn_model is True, dictates whether to implement a
        simplified set of equations with just the synapse (True),
        or to implement the entire postsynaptic set of equations (LHb
        inputs, spiking thresh, AHP, etc.) (False)

    add_pfc : bool
        Add a PFC input to the excitatory LIF dynamics.


    Returns
    --------
    eqs : SimpleNamespace
        eqs.excit_lif_dynamics: A string describing the model for 5-HT
            neurons.
        eqs.syn_ee_model: A string describing the model for 5-HT1AR
            synapses
        eqs.syn_ee_on_pre: A string describing the series of steps
            to occur at a 5-HT1AR synapse when a presynaptic
            spike occurs.
    """

    eqs = SimpleNamespace()

    e = np.e

    # 5-HT1A synapse model
    # ---------
    if new_syn_model is False:
        eqs.excit_lif_dynamics = """
            ds_AMPA_pois/dt = -s_AMPA_pois/tau_AMPA : 1
            ds_AMPA_lhb/dt = -s_AMPA_lhb/tau_AMPA : 1
            ds_GIRK/dt = (-s_GIRK + actfrac_5ht1ar)/tau_GIRK : 1

            ds_ahp_slow/dt = -s_ahp_slow/tau_ahp_slow : 1
            ds_ahp_med/dt = -s_ahp_med/tau_ahp_med : 1
            ds_ahp_fast/dt = -s_ahp_fast/tau_ahp_fast : 1

            dc_5ht/dt = -c_5ht/tau_5ht : mmolar
            actfrac_5ht1ar = (c_5ht**(hill_coeff)) / (ka_1a**(hill_coeff)
                + c_5ht**(hill_coeff)) : 1

            g_lhb_unit_rand : siemens
            g_pois_unit_rand : siemens

            g_lhb = s_AMPA_lhb * g_lhb_unit_rand : siemens
            g_pois = s_AMPA_pois * g_pois_unit_rand : siemens
            g_girk = s_GIRK * g_girk_max : siemens

            g_ahp_slow = s_ahp_slow * g_ahp_slow_unitary : siemens
            g_ahp_med = s_ahp_med * g_ahp_med_unitary : siemens
            g_ahp_fast = s_ahp_fast * g_ahp_fast_unitary : siemens

            I_lhb = g_lhb * (v-E_AMPA) : amp
            I_pois = g_pois * (v-E_AMPA) : amp
            I_girk = g_girk * (v-E_K) : amp

            I_ahp_slow = g_ahp_slow * (v-E_K) : amp
            I_ahp_med = g_ahp_med * (v-E_K) : amp
            I_ahp_fast = g_ahp_fast * (v-E_K) : amp
            I_ahp = I_ahp_slow + I_ahp_med + I_ahp_fast : amp

            I_const : amp

            I_tot = I_lhb + I_pois + I_girk + I_ahp + I_const : amp

            dv/dt =  (
                - G_leak_excit * (v-E_leak_excit)
                - I_tot
                )/Cm_excit : volt (unless refractory)
        """

        eqs.syn_ee_model = """
            pr : 1
            dc_Ca/dt = (-c_Ca+c_Ca0)/tau_Ca : mmolar
        """
        eqs.syn_ee_on_pre = """
            c_Ca+=unit_Ca
            pr = pr_max * (c_Ca**hill_coeff_Ca) \
                / (c_Ca**hill_coeff_Ca + ka_Ca**hill_coeff_Ca)
            c_5ht+=unit_5ht_release*(rand()<pr)
        """
    elif new_syn_model is True:
        if simplify is False:
            if add_pfc is False:
                eqs.excit_lif_dynamics = """
                    ds_AMPA_pois/dt = -s_AMPA_pois/tau_AMPA : 1
                    ds_AMPA_lhb/dt = -s_AMPA_lhb/tau_AMPA : 1

                    s_GIRK = s_GIRK_fall - s_GIRK_rise : 1

                    ds_GIRK_rise/dt = -s_GIRK_rise/tau_GIRK_rise : 1
                    ds_GIRK_fall/dt = -s_GIRK_fall/tau_GIRK_fall : 1

                    ds_ahp_slow/dt = -s_ahp_slow/tau_ahp_slow : 1
                    ds_ahp_med/dt = -s_ahp_med/tau_ahp_med : 1
                    ds_ahp_fast/dt = -s_ahp_fast/tau_ahp_fast : 1

                    g_lhb_unit_rand : siemens
                    g_pois_unit_rand : siemens

                    g_lhb = s_AMPA_lhb * g_lhb_unit_rand : siemens
                    g_pois = s_AMPA_pois * g_pois_unit_rand : siemens
                    g_girk = s_GIRK * g_girk_unit : siemens

                    g_ahp_slow = s_ahp_slow * g_ahp_slow_unitary : siemens
                    g_ahp_med = s_ahp_med * g_ahp_med_unitary : siemens
                    g_ahp_fast = s_ahp_fast * g_ahp_fast_unitary : siemens

                    I_lhb = g_lhb * (v-E_AMPA) : amp
                    I_pois = g_pois * (v-E_AMPA) : amp
                    I_girk = abs(g_girk) * (v-E_K) : amp

                    I_ahp_slow = g_ahp_slow * (v-E_K) : amp
                    I_ahp_med = g_ahp_med * (v-E_K) : amp
                    I_ahp_fast = g_ahp_fast * (v-E_K) : amp
                    I_ahp = I_ahp_slow + I_ahp_med + I_ahp_fast : amp

                    I_const : amp

                    I_tot = I_lhb + I_pois + I_girk + I_ahp + I_const : amp

                    dv/dt =  (
                        - G_leak_excit * (v-E_leak_excit)
                        - I_tot
                        )/Cm_excit : volt (unless refractory)
                """
            elif add_pfc is True:
                eqs.excit_lif_dynamics = """
                    ds_AMPA_pois/dt = -s_AMPA_pois/tau_AMPA : 1
                    ds_AMPA_lhb/dt = -s_AMPA_lhb/tau_AMPA : 1
                    ds_AMPA_pfc/dt = -s_AMPA_pfc/tau_AMPA : 1

                    s_GIRK = s_GIRK_fall - s_GIRK_rise : 1

                    ds_GIRK_rise/dt = -s_GIRK_rise/tau_GIRK_rise : 1
                    ds_GIRK_fall/dt = -s_GIRK_fall/tau_GIRK_fall : 1

                    ds_ahp_slow/dt = -s_ahp_slow/tau_ahp_slow : 1
                    ds_ahp_med/dt = -s_ahp_med/tau_ahp_med : 1
                    ds_ahp_fast/dt = -s_ahp_fast/tau_ahp_fast : 1

                    g_lhb_unit_rand : siemens
                    g_pfc_unit_rand : siemens
                    g_pois_unit_rand : siemens

                    g_lhb = s_AMPA_lhb * g_lhb_unit_rand : siemens
                    g_pfc = s_AMPA_pfc * g_pfc_unit_rand : siemens
                    g_pois = s_AMPA_pois * g_pois_unit_rand : siemens
                    g_girk = s_GIRK * g_girk_unit : siemens

                    g_ahp_slow = s_ahp_slow * g_ahp_slow_unitary : siemens
                    g_ahp_med = s_ahp_med * g_ahp_med_unitary : siemens
                    g_ahp_fast = s_ahp_fast * g_ahp_fast_unitary : siemens

                    I_lhb = g_lhb * (v-E_AMPA) : amp
                    I_pfc = g_pfc * (v-E_AMPA) : amp
                    I_pois = g_pois * (v-E_AMPA) : amp
                    I_girk = abs(g_girk) * (v-E_K) : amp

                    I_ahp_slow = g_ahp_slow * (v-E_K) : amp
                    I_ahp_med = g_ahp_med * (v-E_K) : amp
                    I_ahp_fast = g_ahp_fast * (v-E_K) : amp
                    I_ahp = I_ahp_slow + I_ahp_med + I_ahp_fast : amp

                    I_const : amp

                    I_tot = I_lhb + I_pfc + I_pois + I_girk + I_ahp + I_const : amp

                    dv/dt =  (
                        - G_leak_excit * (v-E_leak_excit)
                        - I_tot
                        )/Cm_excit : volt (unless refractory)
                """

        elif simplify is True:
            eqs.excit_lif_dynamics = """
                ds_GIRK_rise/dt = -s_GIRK_rise/tau_GIRK_rise : 1
                ds_GIRK_fall/dt = -s_GIRK_fall/tau_GIRK_fall : 1

                s_GIRK = s_GIRK_fall - s_GIRK_rise : 1
                g_girk = s_GIRK * g_girk_unit : siemens

                E_driving = v-E_K : volt
                I_girk = g_girk * E_driving : amp

                I_tot = I_girk : amp

                dv/dt =  (
                    - G_leak_excit * (v-E_leak_excit)
                    - I_tot
                    )/Cm_excit : volt (unless refractory)
            """

        # Synapse model and on_pre instructions
        # --------------------
        eqs.syn_ee_model = """
            dstp_sht/dt = -stp_sht/tau_stp_sht : 1 (event-driven)
            dstp_lng_neg/dt = -stp_lng_neg/tau_stp_lng_neg : 1 (event-driven)
            dstp_lng_pos/dt = -stp_lng_pos/tau_stp_lng_pos : 1 (event-driven)

            k = stp_sht*amp_stp_sht
                + (stp_lng_pos-stp_lng_neg)*amp_stp_lng: 1 
            efficacy = 1 / (1 / (1+e**(-1 * b)))
                * (1 / (1 + e**(-1 * (k + b)))) : 1

        """

        eqs.syn_ee_on_pre = """
            stp_sht += 1
            stp_lng_neg += 1
            stp_lng_pos += 1

            s_GIRK_rise += efficacy
            s_GIRK_fall += efficacy

        """

    return eqs


def make_pre_spks(freq, n_pulses, t_start=1):
    """
    Generate a list of presynaptic spiketrain times, given
    a frequency and pulse number.

    Parameters
    ----------
    freq : float
        Frequency (Hz) of the presynaptic spiketrain.
    n_pulses : int
        Number of pulses.
    t_start : float
        Start time of pulsetrain (seconds).

    Returns
    --------
    pre_spks : np.array
        Array of presynaptic spiketimes, in seconds.

    """

    t_end = t_start + (1/freq)*(n_pulses-1) + 0.001
    pre_spks = np.arange(t_start, t_end, 1/freq)*b2.second

    return pre_spks  


def integrate_girk(data):
    """
    Integrates GIRK conductance from a simulated dataset.

    Parameters
    ----------
    data : SimpleNamespace
        Returned from a simulation; has .g_girk as an attribute.

    Returns
    --------
    integ : float
    Integral of GIRK, in nS*second

    """
    integ = np.trapz(data.g_girk.g_girk / b2.nS, t=data.g_girk.t/b2.second)

    return integ


def calc_charge(data, t_start=0*b2.second, t_end=None):
    """
    Returns total charge stats from a simulated dataset, by integrating
    total current over a defined time window.

    Parameters
    ----------
    data : SimpleNamespace
        Returned from a simulation; has .I_tot as an attribute.
    t_start : int (default 0)
        Start time of integration in seconds
    t_end : int (default None)
        End time of integration in seconds. Defaults to the end of recording.

    Returns
    --------
    integ : dict
        integ['all'] : numpy array of charges from each neuron
        integ['mean'] : mean value of integ['all']
        integ['std'] : std of integ['all']
    """
    n_ex_neurs = data.I_tot.I_tot.shape[0]
    integ = {'all': np.empty(n_ex_neurs)}

    ind_t_start = np.argmin(np.abs(data.I_tot.t - t_start))
    if t_end is None:
        ind_t_end = data.I_tot.t.shape[0]
    else:
        ind_t_end = np.argmin(np.abs(data.I_tot.t - t_end))

    for ind_neur in range(n_ex_neurs):
        integ['all'][ind_neur] = np.trapz(
            data.I_tot.t[ind_t_start:ind_t_end],
            data.I_tot.I_tot[ind_neur, ind_t_start:ind_t_end])

    integ['mean'] = np.mean(integ['all'])
    integ['std'] = np.std(integ['all'])

    return integ


def calc_neur_fr(data, ind_neurs, t_start, t_end):
    """
    Returns the individual firing rates of a number of neurons in a particular
    time range

    Parameters
    ----------
    data : SimpleNamespace
        Returned from a simulation; has .spike as an attribute.
    ind_neurs : int or list
        Indices of neurons to return firing rate from
    t_start : float
        Start time of integration in seconds
    t_end : float
        End time of integration in seconds.

    Returns
    --------
        rates : np.ndarray of firing rates
    """

    n_neurs = len(ind_neurs)
    t_dur = t_end - t_start

    rates = np.empty(n_neurs)
    spike_trains = data.spike.spike_trains()

    for ind, ind_neur in enumerate(ind_neurs):
        _spk_tr = spike_trains[ind_neur]
        _n_spks = _spk_tr[(_spk_tr > t_start*b2.second) &
                          (_spk_tr < t_end*b2.second)]
        rates[ind] = len(_n_spks) / t_dur

    return rates


def calc_fr_from_spikemonitor(spikemon,
                              ind_neur_min=None,
                              ind_neur_max=None,
                              dt=100*b2.ms,
                              t_end=None):
    if t_end is None:
        t_end = np.max(spikemon.t)*b2.second

    fr = SimpleNamespace()
    fr.t = np.arange(t_end, step=dt)*b2.second
    fr.rate = np.empty(len(fr.t))

    # If inds_neur is not given, set to all neurs
    if ind_neur_min is None:
        ind_neur_min = 0
    if ind_neur_max is None:
        ind_neur_max = np.max(spikemon.i)

    n_neurs = ind_neur_max - ind_neur_min

    # Iteratively calculate firing rate
    inds_neur_in_spkmon = np.logical_and(
        spikemon.i > ind_neur_min,
        spikemon.i < ind_neur_max)

    for ind, _t in enumerate(fr.t):
        _t_min = _t
        _t_max = _t + dt

        _inds_t_in_spkmon = np.logical_and(
            spikemon.t[inds_neur_in_spkmon] > _t_min,
            spikemon.t[inds_neur_in_spkmon] < _t_max)

        fr.rate[ind] = sum(_inds_t_in_spkmon) / (n_neurs * dt)

    return fr
