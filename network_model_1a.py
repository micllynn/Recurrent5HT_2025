# -*- coding: utf-8 -*-
"""
@author: michaellynn
Use with env: brian2
"""

import os

import numpy as np
from types import SimpleNamespace

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import brian2 as b2
from brian2 import NeuronGroup, Synapses, PoissonInput, PoissonGroup, \
    SpikeGeneratorGroup, prefs
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor

from helper_fns import make_eqs, make_pre_spks, calc_neur_fr, \
    calc_fr_from_spikemonitor

# brian2 runtime parameters
prefs.codegen.target = 'cython'
b2.defaultclock.dt = 0.1 * b2.ms

# simulations of single synapses (to observe STP dynamics)
# ------------------


def simple_sim(stp_lin_nonlin=True,
               pre_spks=make_pre_spks(freq=20, n_pulses=2),
               mon_dt=1 * b2.ms,
               integ_dt=0.1*b2.ms,
               ni_method='rk2',
               g_girk_unit=0.1*b2.nS,
               tau_GIRK_fall=308.04 * b2.ms,
               tau_GIRK_rise=85.97 * b2.ms,
               tau_stp_sht=127.89 * b2.ms,
               tau_stp_lng_pos=1266.14 * b2.ms,
               tau_stp_lng_neg=698.48 * b2.ms,
               amp_stp_sht=1.174,
               amp_stp_lng=-1.798,
               b=-6.223,
               plot=False,
               fig_str='20Hz'):
    """
    Runs a simple simulation of the 5-HT1AR synapss, and returns a
    SimpleNamespace of results.

    Allows for gradient descent on GIRK kernel and STP kernel, as well as
    STP nonlinearity, by comparing simulation to experiment.

    Parameters
    ----------
    stp_lin_nonlin : bool
        Whether to use linear-nonlinear model of STP (True), or
        a regular Ca2+ accumulation + Hill equation model (False).
    pre_spks : np.array
        Array of presynaptic spiketimes, in seconds.
    mon_dt : float * b2.ms
        Monitor sampling rate
    g_girk_max : float* b2.nS
        Maximum GIRK conductance
    tau_GIRK_fall : float * b2.ms
        Tau of falling phase of girk
    tau_GIRK_rise : float * b2.ms
        Tau of rising phase of girk (subtractive filter)
    tau_stp_sht : float * b2.ms
        Tau of short-timescale STF
    tau_stp_lng_neg : float * b2.ms
        Tau of long-timescale STD
    tau_stp_lng_pos : float * b2.ms
        Tau of negative polarity filter cancelling out the early
        component of long-timescale STD
    amp_stp_sht : float
        Amplitude (arbitrary units) of short STF
    amp_stp_lng : float
        Amplitude (arbitrary units) of long STD
    b : float
        Baseline term in STP nonlinearity
    plot : boolean
        Whether to generate a plot of the result
    fig_str : str
        String appended to end of figure name during save

    Returns
    --------
    data : SimpleNamespace
        data.pre_spk : SpikeMonitor for presynaptic barrage
        data.syn_kern : StateMonitor for synaptic kernel
        data.syn_effic : StateMonitor for synaptic efficacy
        data.g_girk : StateMonitor for GIRK conductance
        data.I_tot : StateMonitor for total current
    """
    b2.defaultclock.dt = integ_dt

    print('\tInitializing parameters...')
    t_sim = pre_spks[-1] + 3*b2.second

    # Shared variables
    E_K = -90 * b2.mV  # reversal pot. for potassium

    # Serotonon neuron parameters
    # ---------------------
    Cm_excit = 0.0876 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 1.4 * b2.nS  # leak conductance
    E_leak_excit = -68.07 * b2.mV  # reversal potential
    v_firing_threshold_excit = -52.33 * b2.mV  # spike condition
    v_reset_excit = -70.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 6.5 * b2.ms  # absolute refractory period

    # Generate equations for the synapses
    # -----------------------
    eqs = make_eqs(new_syn_model=stp_lin_nonlin, simplify=True)

    # Neuron groups
    # *******************************************************
    # Postsynaptic neuron
    # -------------------
    excit_pop = NeuronGroup(1,
                            model=eqs.excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit",
                            reset='''v=v_reset_excit''',
                            refractory=t_abs_refract_excit,
                            events={'voltage_floor':  'v < -89.99*mV'},
                            method=ni_method)

    excit_pop.run_on_event('voltage_floor', 'v = -89.99*mV')

    # initialize with random voltages:
    excit_pop.v = -70 * b2.mV

    # Presynaptic neuron
    # --------------------
    pre_neur = SpikeGeneratorGroup(1, np.zeros(len(pre_spks)),
                                   pre_spks)

    # Connections
    # *****************************************************************
    # 5-HT-->5-HT
    # ----------
    syn_excit_excit = Synapses(pre_neur,
                               target=excit_pop,
                               model=eqs.syn_ee_model,
                               on_pre=eqs.syn_ee_on_pre,
                               method="rk2")
    syn_excit_excit.connect(i=[0], j=[0])

    # brian2 monitors
    # *****************************************************************
    print('\tSetting up monitors...')

    # presynaptic spk monitors
    mon_pre_spk = SpikeMonitor(pre_neur)

    # synaptic monitors
    mon_syn_kern = StateMonitor(syn_excit_excit,
                                'k',
                                record=True,
                                dt=mon_dt)
    mon_syn_effic = StateMonitor(syn_excit_excit,
                                 'efficacy',
                                 record=True,
                                 dt=mon_dt)


    # 5-HT cell monitors
    mon_g_girk = StateMonitor(excit_pop,
                              'g_girk',
                              record=True,
                              dt=mon_dt)
    mon_E_driving = StateMonitor(excit_pop,
                                 'E_driving',
                                 record=True,
                                 dt=mon_dt)
    mon_I_tot = StateMonitor(excit_pop,
                             'I_tot',
                             record=True,
                             dt=mon_dt)
    mon_v = StateMonitor(excit_pop,
                         'v',
                         record=True,
                         dt=mon_dt)
    mon_spk = SpikeMonitor(excit_pop,
                           record=True)

    # Gather and run
    # *******************************************************************
    print('\tGathering network objects...')
    net = b2.Network(pre_neur, excit_pop, syn_excit_excit,
                     mon_pre_spk, mon_syn_kern, mon_syn_effic,
                     mon_g_girk, mon_E_driving, mon_I_tot, mon_v, mon_spk)

    print('\t\tRunning network simulation...\n')
    net.run(t_sim, report='stdout', report_period=500*b2.ms)

    # Save variables
    # ******************************************************************
    data = SimpleNamespace(pre_spk=mon_pre_spk,
                           syn_kern=mon_syn_kern,
                           syn_effic=mon_syn_effic,
                           g_girk=mon_g_girk,
                           E_driving=mon_E_driving,
                           I_tot=mon_I_tot,
                           v=mon_v, spk=mon_spk)

    # Plot
    # ****************************************
    if plot is True:
        # plt 1: STP kernel
        t_post = np.arange(0, 4, 0.01)
        t_pre = np.arange(-1, 0, 0.01)
        t = np.append(t_pre, t_post)

        kern_post = (amp_stp_sht*np.exp(-t_post/(tau_stp_sht/b2.second))
                     + amp_stp_lng*(
                         np.exp(-t_post/(tau_stp_lng_pos/b2.second)) -
                         np.exp(-t_post/(tau_stp_lng_neg/b2.second))))
        kern_pre = np.zeros_like(t_pre)
        kern = np.append(kern_pre, kern_post)

        fig0 = plt.figure(figsize=(2, 2), constrained_layout=True)
        ax_kern = fig0.add_subplot(1, 1, 1)
        ax_kern.plot(t, kern)
        ax_kern.set_xlabel('time (s)')
        ax_kern.set_ylabel('$k_{\mu}$')
        fig0.savefig('kernel.pdf')

        # Plt 2: all synaptic variables
        plt.style.use('publication_ml')
        fig = plt.figure(figsize=(4, 4), constrained_layout=True)
        spec = gridspec.GridSpec(nrows=4,
                                 ncols=1,
                                 figure=fig,
                                 height_ratios=[1, 1, 1, 1])

        ax_girk = fig.add_subplot(spec[3, 0])
        ax_pre_spks = fig.add_subplot(spec[0, 0], sharex=ax_girk)
        ax_syn_kern = fig.add_subplot(spec[1, 0], sharex=ax_girk)
        ax_syn_effic = fig.add_subplot(spec[2, 0], sharex=ax_girk)

        list_ax = [ax_pre_spks, ax_syn_kern, ax_syn_effic, ax_girk]

        for spk in data.pre_spk.t:
            ax_pre_spks.plot([spk/b2.second, spk/b2.second],
                             [0, 1], 'k')
        ax_pre_spks.set_ylabel('$spk_{pre}$')

        ax_syn_kern.plot(data.syn_kern.t/b2.second,
                         data.syn_kern.k[0, :], 'k')
        ax_syn_kern.set_ylabel('$k$')

        ax_syn_effic.plot(data.syn_effic.t/b2.second,
                          data.syn_effic.efficacy[0, :], 'k')
        ax_syn_effic.set_ylabel('$\mu$')

        ax_girk.plot(data.g_girk.t/b2.second,
                     data.g_girk.g_girk[0, :]/b2.nS, 'k')
        ax_girk.set_ylabel('$g_{girk}$ (nS)')

        ax_girk.set_xlabel('time (s)')

        fig.savefig(f'figs/simple_sim_{fig_str}.pdf')
        plt.show()

    return data


def plot_simple_sims(freq_conds=[1, 20],
                     n_pulses=8):
    """
    Plots the result of simple_sim().
    """

    # Simulate
    # -------------
    data = {}

    for freq_cond in freq_conds:
        data[str(freq_cond)] = simple_sim(
            pre_spks=make_pre_spks(freq=freq_cond, n_pulses=n_pulses),
            plot=False)

    # Plot
    # ------------
    colors={'1': sns.xkcd_rgb['black'],
            '20': sns.xkcd_rgb['cornflower']}

    plt.style.use('publication_ml')
    fig = plt.figure(figsize=(3.43, 3.43))
    spec = gridspec.GridSpec(nrows=4,
                             ncols=1,
                             figure=fig,
                             height_ratios=[1, 1, 1, 1])

    ax_girk = fig.add_subplot(spec[3, 0])
    ax_pre_spks = fig.add_subplot(spec[0, 0], sharex=ax_girk)
    ax_syn_kern = fig.add_subplot(spec[1, 0], sharex=ax_girk)
    ax_syn_effic = fig.add_subplot(spec[2, 0], sharex=ax_girk)

    list_ax = [ax_pre_spks, ax_syn_kern, ax_syn_effic, ax_girk]

    for freq_cond in freq_conds:
        freq_cond_str = str(freq_cond)
        _data = data[freq_cond_str]

        for spk in _data.pre_spk.t:
            ax_pre_spks.plot([spk/b2.second, spk/b2.second],
                             [0, 1], color=colors[freq_cond_str])
        ax_pre_spks.set_ylabel('$spk_{pre}$')

        ax_syn_kern.plot(_data.syn_kern.t/b2.second,
                         _data.syn_kern.k[0, :],
                         color=colors[freq_cond_str])
        ax_syn_kern.set_ylabel('$k$')

        ax_syn_effic.plot(_data.syn_effic.t/b2.second,
                          _data.syn_effic.efficacy[0, :],
                          color=colors[freq_cond_str])
        ax_syn_effic.set_ylabel('$\mu$')

        ax_girk.plot(_data.g_girk.t/b2.second,
                     _data.g_girk.g_girk[0, :]/b2.nS,
                     color=colors[freq_cond_str])
        ax_girk.set_ylabel('$g_{girk}$ (nS)')

        ax_girk.set_xlabel('time (s)')

    plt.savefig(f'figs/simple_sims_overlay.pdf')
    plt.show()

    return

# simple plots showing parameterization of 5-ht connection distance
# --------------------------


def test_ee_pconn(std=1000, n_neurs=1000, p_conn_max=0.005):
    """
    Plots recurrent connection probability given a connection stdev.
    """
    i = np.arange(-1*n_neurs, n_neurs)
    p = p_conn_max * np.exp(-1*(i**2)/(2*std**2))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(i, p)
    plt.show()
    fig.savefig(os.path.join('figs', 'pconn_fig_single.pdf'))

    return


def test_ee_pconn_multiplot(stds=[1250, 2500, 5000], n_neurs=5000,
                            p_conn_max=0.005):
    """
    Plots many recurrent connection probabilities given
    a list of stdevs.
    """
    colors = sns.cubehelix_palette(start=2, rot=0,
                                   dark=0.4, light=0.8,
                                   n_colors=len(stds))

    i = np.arange(-1*(n_neurs/2), n_neurs/2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for ind, std in enumerate(stds):
        _p = p_conn_max * np.exp(-1*(i**2)/(2*std**2))
        ax.plot(i, _p, color=colors[ind], label=f'stdev={std/n_neurs}')
    ax.set_xlabel('5-HT neuron')
    ax.set_ylabel('p(connect)')

    plt.show()
    fig.savefig(os.path.join('figs', 'pconn_fig.pdf'))

    return


# main network simulations
# --------------------------


def sim(sim_time=4*b2.second,
        sim_dt=0.1*b2.ms,
        mon_dt=5*b2.ms,
        n_5ht=5000,
        n_lhb=500,
        n_neur_samp=100,
        poisson_rate=25,
        I_const_mean=10,
        I_const_std=5,
        p_lhb_input=0.25,
        p_connect_lhb_drn=0.01,
        dict_ee_connect={'p': 0.05},
        autoinhib=False,
        lhb_ampli=5,
        t_lhb_tot=400,
        t_lhb_delay=2000,
        tau_GIRK_fall=308.04 * b2.ms,
        tau_GIRK_rise=85.97 * b2.ms,
        tau_stp_sht=127.89 * b2.ms,
        tau_stp_lng_pos=1266.14 * b2.ms,
        tau_stp_lng_neg=698.48 * b2.ms,
        amp_stp_sht=1.174,
        amp_stp_lng=-1.798,
        b=-6.223,
        g_girk_unit=0.01*b2.nS,
        g_pois_unit=0.5*b2.nS,
        g_pois_std=0.1*b2.nS,
        g_lhb_unit=1.5*b2.nS,
        g_lhb_std=0.2*b2.nS):

    """
    Function runs a simulation of a network of recurrently connected
    LIF 5-HT neurons (DRN). Recurrent connections are treated with
    the SRP model (short-term plasticity).

    Network receives input from LHb as well as nonspecific poisson input.

    Parameters
    -------------------------------------
    *** Basic parameters ***
    sim_time : float
        Total time to simulate (ms)
    mon_dt : float
        The sampling interval for the brian2 monitors (I, g, etc.).
        Recommended to use a larger interval for space conservation.
        (In ms)
    n_5ht : int
        number of 5-HT neurons in DRN
    n_lhb : int
        Number of lhb neurons providing input
    n_neur_samp : int
        Number of neurons to sample in the brian2 monitors for export.
        (If it's too big, scripts may be memory-limited over many iters.)
    poisson_rate : float
        Rate of background poisson input to DRN (Hz)
    I_const_mean : float
        Mean of constant-current input for each 5-HT neuron (pA)
        (Used to construct normal distribution from which I_const
        is assigned for each neuron)

    *** Connectivity and LHb inputs ***
    I_const_std : float
        Standard dev. of constant-current input for each 5-HT neuron (pA)
    p_lhb_input : float
        The fraction of 5-HT neurons which will be targeted for random
        (non-zero) connectivity with LHb.
    p_connect_lhb_drn : float
        Connection probability between each LHb neuron and each LHb-
        recipient DRN neuron
    p_connect_e_e : float
        Connection probability between each DRN neuron and itself
        (5-HT1A synapse)
            For distance-dependent connectivity:
            dict_ee_connect={'p': '0.005*exp((-1*(i-j)**2)/(2*std**2))'}
                (where std is std dev of gaussian connectivity, in neurons)
    autoinhib : boolean (default False)
        If True, models 1A synapses as only autoinhibitory (eg self-to-self).
        This provides a means to mimic AHPs using an identical conductance
        load as the typical 1A conductance. Changes:
            A. Connection probability is updated (autosynapses only).
            B: GIRK conductance is updated (scaled by p_connect_e_e by default,
                to preserve conductances for an equivalent network)
    lhb_ampli : float
        Firing rate of LHb neurons providing input (Hz)
    t_lhb_tot : float
        The total duration over which LHb fires at lhb_ampli (ms)
    t_lhb_delay : float
        They delay from the start of sim to when LHb begins to
        fire at lhb_ampli (ms)

    *** Presynaptic facilitation (5-HT) ***
    (All terms described in Eqn 8 of Lynn et al (2025);
    SRP model of synaptic plasticity)

    tau_stp_sht : float
        value for tau_stp_sht
    tau_stp_lng_pos : float
        value for tau_stp_lng_pos
    tau_stp_lng_neg : float
        value for tau_stp_lng_neg
    amp_stp_sht : float
        value for amp_stp_sht
    amp_stp_lng : float
        value for amp_stp_lng
    b : float
        value for b

    *** Conductances ***
    g_pois_unit : float
        Mean of unitary conductance for poisson input to 5-HT neurs (nS)
    g_pois_std : float
        St dev of unitary conductance for pois input to 5-HT neurs (nS)
    g_lhb_unit : float
        Mean of unit conduct for lhb input to 5-HT neurs (nS)
    g_lhb_std : float
        St dev of unit conduct for lhb input to 5-HT neurs (nS)
    g_girk_max : float
        Maximal GIRK conductance for each cell (nS).

    Returns
    -----------
    data : SimpleNamespace
        This SimpleNamespace stores all brian2 monitors
        from the simulation. It is equivalent to the following:

        data = SimpleNamespace(spike=mon_spike,
                               rate_5ht=mon_5ht_rate,
                               rate_lhb=mon_lhb_rate,
                               v=mon_v,
                               syn_k=mon_syn_kern,
                               syn_effic=mon_syn_effic,
                               g_girk=mon_g_girk,
                               I_tot=mon_I_tot)
    """

    b2.defaultclock.dt = sim_dt

    print('\tInitializing parameters...')
    # Shared variables
    E_AMPA = 0.0 * b2.mV  # Veq for A/N
    E_K = -90.0 * b2.mV  # reversal pot. for potassium

    # Serotonon neuron parameters
    # ---------------------
    Cm_excit = 0.0876 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 1.4 * b2.nS  # leak conductance
    E_leak_excit = -68.07 * b2.mV  # reversal potential
    v_firing_threshold_excit = -52.33 * b2.mV  # spike condition
    v_reset_excit = -68.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 6.5 * b2.ms  # absolute refractory period

    # Serotonin ahp params
    # --------------------
    # Calculated based on reported mean I_ahp values from
    # Emerson Harkin's GIF model.
    tau_ahp_fast = 3.0 * b2.ms
    tau_ahp_med = 30.0 * b2.ms
    tau_ahp_slow = 300 * b2.ms

    _targ_I_ahp_fast = 0.0281 * b2.nA
    _targ_I_ahp_med = 0.0333 * b2.nA
    _targ_I_ahp_slow = 0.0210 * b2.nA

    g_ahp_fast_unitary = _targ_I_ahp_fast / (v_firing_threshold_excit - E_K)
    g_ahp_med_unitary = _targ_I_ahp_med / (v_firing_threshold_excit - E_K)
    g_ahp_slow_unitary = _targ_I_ahp_fast / (v_firing_threshold_excit - E_K)

    # Excit. synapses
    # ---------
    # For LHb/poisson input; really a combo of AMPA/NMDA
    tau_AMPA = 20.0 * b2.ms  # Tau reflecting combo of AMPA/NMDA

    # Serotonin neuron LIF model
    # ---------
    eqs = make_eqs(new_syn_model=True, simplify=False)

    # Neuron groups
    # *******************************************************
    # Postsynaptic neuron
    # -------------------
    excit_pop = NeuronGroup(n_5ht,
                            model=eqs.excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit",
                            reset='''v=v_reset_excit
                            s_ahp_slow+=1
                            s_ahp_med+=1
                            s_ahp_fast+=1''',
                            refractory=t_abs_refract_excit,
                            events={'voltage_floor':  'v < -90.0*mV'},
                            method='rk2')

    excit_pop.run_on_event('voltage_floor', 'v = -89.9*mV')

    # initialize with random voltages:
    excit_pop.v = np.random.normal(loc=(v_reset_excit + 0.5 * b2.mV) / b2.mV,
                                   scale=5,
                                   size=n_5ht) * b2.mV
    excit_pop.I_const = np.random.normal(loc=I_const_mean*-1,
                                         scale=I_const_std,
                                         size=n_5ht) * b2.pA

    # Initialize
    excit_pop.g_lhb_unit_rand = np.random.normal(loc=(g_lhb_unit/b2.nS),
                                                 scale=g_lhb_std/b2.nS,
                                                 size=n_5ht) * b2.nS
    excit_pop.g_lhb_unit_rand[excit_pop.g_lhb_unit_rand < 0] = 0

    excit_pop.g_pois_unit_rand = np.random.normal(loc=(g_pois_unit/b2.nS),
                                                  scale=g_pois_std/b2.nS,
                                                  size=n_5ht) * b2.nS
    excit_pop.g_pois_unit_rand[excit_pop.g_pois_unit_rand < 0] = 0
    # Connections
    # *****************************************************************
    # poisson-->5-HT
    # ----------
    poisson = PoissonInput(excit_pop,
                           's_AMPA_pois',
                           1,
                           poisson_rate * b2.Hz,
                           weight=1)

    # LHb-->5-HT
    # ----------
    str_ratecond_lhb = f"""lhb_ampli*(t>{t_lhb_delay}*ms \
        and t<({t_lhb_delay}+{t_lhb_tot})*ms)*Hz"""
    lhb_pop = PoissonGroup(n_lhb, str_ratecond_lhb)
    syn_lhb_drn = Synapses(lhb_pop, excit_pop[0:int(n_5ht*p_lhb_input)],
                           on_pre="s_AMPA_lhb += 1")
    syn_lhb_drn.connect(p=p_connect_lhb_drn)

    # 5-HT-->5-HT
    # ----------
    syn_excit_excit = Synapses(excit_pop,
                               target=excit_pop,
                               model=eqs.syn_ee_model,
                               on_pre=eqs.syn_ee_on_pre,
                               method="rk2")

    if autoinhib is True:
        unity_vec = np.arange(len(excit_pop))
        syn_excit_excit.connect(i=unity_vec, j=unity_vec)
    elif autoinhib is False:
        syn_excit_excit.connect(**dict_ee_connect)

    # brian2 monitors
    # *****************************************************************
    print('\tSetting up spike/voltage monitors...')

    # General
    mon_spike = SpikeMonitor(excit_pop)
    mon_5ht_rate = PopulationRateMonitor(excit_pop)
    mon_lhb_rate = PopulationRateMonitor(lhb_pop)

    # Ca, pr monitors
    # Pick random synapses from synapse population since the inds
    # don't exactly map onto neuron numbers here.
    ind_syns = np.linspace(0, n_5ht, n_neur_samp).astype(int)

    # synaptic monitors
    mon_syn_kern = StateMonitor(syn_excit_excit,
                                'k',
                                record=ind_syns,
                                dt=mon_dt)
    mon_syn_effic = StateMonitor(syn_excit_excit,
                                 'efficacy',
                                 record=ind_syns,
                                 dt=mon_dt)

    # 5-HT cell monitors
    inds_record = np.sort((np.random.rand(n_neur_samp)*n_5ht).astype(int))

    mon_v = StateMonitor(excit_pop,
                         'v',
                         record=inds_record,
                         dt=mon_dt)
    mon_g_girk = StateMonitor(excit_pop,
                              'g_girk',
                              record=inds_record,
                              dt=mon_dt)
    mon_I_tot = StateMonitor(excit_pop,
                             'I_tot',
                             record=inds_record,
                             dt=mon_dt)

    # Gather and run
    # *******************************************************************
    print('\tGathering network objects...')
    net = b2.Network(excit_pop, poisson, lhb_pop, syn_lhb_drn, syn_excit_excit,
                     mon_spike, mon_5ht_rate, mon_lhb_rate, mon_v,
                     mon_syn_kern, mon_syn_effic,
                     mon_g_girk, mon_I_tot)

    print('\t\tRunning network simulation...\n')
    net.run(sim_time, report='stdout', report_period=500*b2.ms)

    # Save variables
    # ******************************************************************
    data = SimpleNamespace(spike=mon_spike,
                           rate_5ht=mon_5ht_rate,
                           rate_lhb=mon_lhb_rate,
                           v=mon_v,
                           syn_k=mon_syn_kern,
                           syn_effic=mon_syn_effic,
                           g_girk=mon_g_girk,
                           I_tot=mon_I_tot)
    return data


def sim_lhbpfc(sim_time=3.5*b2.second,
               sim_dt=0.1*b2.ms,
               mon_dt=5*b2.ms,
               n_5ht=5000,
               n_lhb=500,
               n_pfc=500,
               n_neur_samp=100,
               poisson_rate=25,
               I_const_mean=10,
               I_const_std=5,
               p_lhb_input=0.25,
               p_pfc_input=0.25,
               overlap_lhbpfc=0,
               p_connect_lhb_drn=0.01,
               p_connect_pfc_drn=0.01,
               dict_ee_connect={'p': 0.005},
               autoinhib=False,
               lhb_ampli=5,
               pfc_ampli=0,
               t_lhb_tot=500,
               t_lhb_delay=2500,
               t_pfc_tot=1000,
               t_pfc_delay=2000,
               tau_GIRK_fall=308.04 * b2.ms,
               tau_GIRK_rise=85.97 * b2.ms,
               tau_stp_sht=127.89 * b2.ms,
               tau_stp_lng_pos=1266.14 * b2.ms,
               tau_stp_lng_neg=698.48 * b2.ms,
               amp_stp_sht=1.174,
               amp_stp_lng=-1.798,
               b=-6.223,
               g_girk_unit=0.1*b2.nS,
               g_pois_unit=0.5*b2.nS,
               g_pois_std=0.1*b2.nS,
               g_lhb_unit=1.5*b2.nS,
               g_lhb_std=0.2*b2.nS,
               g_pfc_unit=1.5*b2.nS,
               g_pfc_std=0.2*b2.nS):
    """
    Function runs a simulation of a network of recurrently connected
    LIF 5-HT neurons (DRN). Recurrent connections are treated with
    the SRP model (short-term plasticity).

    Network receives input from LHb, another excitatory area A (here
    called PFC, but not necessarily) and a nonspecific poisson input.

    Parameters
    -------------------------------------
    *** Basic parameters ***
    sim_time : float
        Total time to simulate (ms)
    mon_dt : float
        The sampling interval for the brian2 monitors (I, g, etc.).
        Recommended to use a larger interval for space conservation.
        (In ms)
    n_5ht : int
        number of 5-HT neurons in DRN
    n_lhb : int
        Number of lhb neurons providing input
    n_neur_samp : int
        Number of neurons to sample in the brian2 monitors for export.
        (If it's too big, scripts may be memory-limited over many iters.
    poisson_rate : float
        Rate of background poisson input to DRN (Hz)
    I_const_mean : float
        Mean of constant-current input for each 5-HT neuron (pA)
        (Used to construct normal distribution from which I_const
        is assigned for each neuron)

    *** Connectivity and LHb inputs ***
    I_const_std : float
        Standard dev. of constant-current input for each 5-HT neuron (pA)
    p_lhb_input : float
        The fraction of 5-HT neurons which will be targeted for random
        (non-zero) connectivity with LHb.
    p_pfc_input : float
        The fraction of 5-HT neurons which will be targeted for random
        (non-zero) connectivity with PFC.
    overlap_lhbpfc : float
        Fraction overlap between 5-HT neurons receiving lhb and pfc input
        (Ex: if p_lhb_input=p_pfc_input=0.25, and overlap_lhbpfc=0.5,
        then 0.125 of 5-ht neurons receive only pfc input, 0.125 receive both
        lhb and pfc input, and 0.125 receive only lhb input.)
    p_connect_lhb_drn : float
        Connection probability between each LHb neuron and each LHb-
        recipient DRN neuron
    p_connect_e_e : float
        Connection probability between each DRN neuron and itself
        (5-HT1A synapse)
    autoinhib : boolean (default False)
        If True, models 1A synapses as only autoinhibitory (eg self-to-self).
        This provides a means to mimic AHPs using an identical conductance
        load as the typical 1A conductance. Changes:
            A. Connection probability is updated (autosynapses only).
            B: GIRK conductance is updated (scaled by p_connect_e_e by default,
                to preserve conductances for an equivalent network)
    lhb_ampli : float
        Firing rate of LHb neurons providing input (Hz)
    t_lhb_tot : float
        The total duration over which LHb fires at lhb_ampli (ms)
    t_lhb_delay : float
        They delay from the start of sim to when LHb begins to
        fire at lhb_ampli (ms)

    *** Presynaptic facilitation (5-HT) ***
    (All terms described in Eqn 8 of Lynn et al (2025);
    SRP model of synaptic plasticity)

    tau_stp_sht : float
        value for tau_stp_sht
    tau_stp_lng_pos : float
        value for tau_stp_lng_pos
    tau_stp_lng_neg : float
        value for tau_stp_lng_neg
    amp_stp_sht : float
        value for amp_stp_sht
    amp_stp_lng : float
        value for amp_stp_lng
    b : float
        value for b

    *** Conductances ***
    g_pois_unit : float
        Mean of unitary conductance for poisson input to 5-HT neurs (nS)
    g_pois_std : float
        St dev of unitary conductance for pois input to 5-HT neurs (nS)
    g_lhb_unit : float
        Mean of unit conduct for lhb input to 5-HT neurs (nS)
    g_lhb_std : float
        St dev of unit conduct for lhb input to 5-HT neurs (nS)
    g_girk_max : float
        Maximal GIRK conductance for each cell (nS).

    Returns
    -----------
    data : SimpleNamespace
        This SimpleNamespace stores all brian2 monitors
        from the simulation. It is equivalent to the following:

        data = SimpleNamespace(spike=mon_spike,
                               rate_5ht=mon_5ht_rate,
                               rate_lhb=mon_lhb_rate,
                               v=mon_v,
                               syn_k=mon_syn_kern,
                               syn_effic=mon_syn_effic,
                               g_girk=mon_g_girk,
                               I_tot=mon_I_tot)
    """

    b2.defaultclock.dt = sim_dt

    print('\tInitializing parameters...')
    # Shared variables
    E_AMPA = 0.0 * b2.mV  # Veq for A/N
    E_K = -90.0 * b2.mV  # reversal pot. for potassium

    # Serotonon neuron parameters
    # ---------------------
    Cm_excit = 0.0876 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 1.4 * b2.nS  # leak conductance
    E_leak_excit = -68.07 * b2.mV  # reversal potential
    v_firing_threshold_excit = -52.33 * b2.mV  # spike condition
    v_reset_excit = -68.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 6.5 * b2.ms  # absolute refractory period

    # Serotonin ahp params
    # --------------------
    # Calculated based on reported mean I_ahp values from
    # Emerson Harkin's GIF model.
    tau_ahp_fast = 3.0 * b2.ms
    tau_ahp_med = 30.0 * b2.ms
    tau_ahp_slow = 300 * b2.ms

    _targ_I_ahp_fast = 0.0281 * b2.nA
    _targ_I_ahp_med = 0.0333 * b2.nA
    _targ_I_ahp_slow = 0.0210 * b2.nA

    g_ahp_fast_unitary = _targ_I_ahp_fast / (v_firing_threshold_excit - E_K)
    g_ahp_med_unitary = _targ_I_ahp_med / (v_firing_threshold_excit - E_K)
    g_ahp_slow_unitary = _targ_I_ahp_fast / (v_firing_threshold_excit - E_K)

    # Excit. synapses
    # ---------
    # For LHb/poisson input; really a combo of AMPA/NMDA
    tau_AMPA = 20.0 * b2.ms  # Tau reflecting combo of AMPA/NMDA

    # Serotonin neuron LIF model
    # ---------
    eqs = make_eqs(new_syn_model=True, simplify=False,
                   add_pfc=True)

    # Neuron groups
    # *******************************************************
    # Postsynaptic neuron
    # -------------------
    excit_pop = NeuronGroup(n_5ht,
                            model=eqs.excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit",
                            reset='''v=v_reset_excit
                            s_ahp_slow+=1
                            s_ahp_med+=1
                            s_ahp_fast+=1''',
                            refractory=t_abs_refract_excit,
                            events={'voltage_floor':  'v < -90.0*mV'},
                            method='rk2')

    excit_pop.run_on_event('voltage_floor', 'v = -90*mV')

    # initialize with random voltages:
    excit_pop.v = np.random.normal(loc=(v_reset_excit + 0.5 * b2.mV) / b2.mV,
                                   scale=5,
                                   size=n_5ht) * b2.mV
    excit_pop.I_const = np.random.normal(loc=I_const_mean*-1,
                                         scale=I_const_std,
                                         size=n_5ht) * b2.pA

    # Initialize
    excit_pop.g_lhb_unit_rand = np.random.normal(loc=(g_lhb_unit/b2.nS),
                                                 scale=g_lhb_std/b2.nS,
                                                 size=n_5ht) * b2.nS
    excit_pop.g_lhb_unit_rand[excit_pop.g_lhb_unit_rand < 0] = 0

    excit_pop.g_pfc_unit_rand = np.random.normal(loc=(g_pfc_unit/b2.nS),
                                                 scale=g_pfc_std/b2.nS,
                                                 size=n_5ht) * b2.nS
    excit_pop.g_pfc_unit_rand[excit_pop.g_pfc_unit_rand < 0] = 0

    excit_pop.g_pois_unit_rand = np.random.normal(loc=(g_pois_unit/b2.nS),
                                                  scale=g_pois_std/b2.nS,
                                                  size=n_5ht) * b2.nS
    excit_pop.g_pois_unit_rand[excit_pop.g_pois_unit_rand < 0] = 0

    # Connections
    # *****************************************************************
    # poisson-->5-HT
    # ----------
    poisson = PoissonInput(excit_pop,
                           's_AMPA_pois',
                           1,
                           poisson_rate * b2.Hz,
                           weight=1)

    # LHb-->5-HT
    # ----------
    str_ratecond_lhb = f"""lhb_ampli*(t>{t_lhb_delay}*ms \
        and t<({t_lhb_delay}+{t_lhb_tot})*ms)*Hz"""
    lhb_pop = PoissonGroup(n_lhb, str_ratecond_lhb)
    n_5ht_rec_lhb = int(n_5ht*p_lhb_input)
    syn_lhb_drn = Synapses(lhb_pop, excit_pop[0:n_5ht_rec_lhb],
                           on_pre="s_AMPA_lhb += 1")
    syn_lhb_drn.connect(p=p_connect_lhb_drn)

    # PFC-->5-HT
    # ----------
    # str_ratecond_pfc = f"""pfc_ampli*(t>{t_pfc_delay}*ms \
    #     and t<({t_pfc_delay}+{t_pfc_tot})*ms)*Hz"""
    str_ratecond_pfc = f"""pfc_ampli*((t-{t_pfc_delay}*ms)/({t_pfc_tot}*ms))* \
    (t>{t_pfc_delay}*ms and t<({t_pfc_delay}+{t_pfc_tot})*ms)*Hz
    """
    pfc_pop = PoissonGroup(n_pfc, str_ratecond_pfc)
    n_5ht_rec_pfc = int(n_5ht*p_pfc_input)

    ind_start_pfc_drn = int(n_5ht_rec_lhb
                            - (overlap_lhbpfc*n_5ht_rec_lhb))
    syn_pfc_drn = Synapses(pfc_pop, excit_pop[
        ind_start_pfc_drn:n_5ht_rec_lhb+n_5ht_rec_pfc],
                           on_pre="s_AMPA_pfc += 1")
    syn_pfc_drn.connect(p=p_connect_pfc_drn)

    # 5-HT-->5-HT
    # ----------
    syn_excit_excit = Synapses(excit_pop,
                               target=excit_pop,
                               model=eqs.syn_ee_model,
                               on_pre=eqs.syn_ee_on_pre,
                               method="rk2")

    if autoinhib is True:
        unity_vec = np.arange(len(excit_pop))
        syn_excit_excit.connect(i=unity_vec, j=unity_vec)
    elif autoinhib is False:
        syn_excit_excit.connect(**dict_ee_connect)

    # brian2 monitors
    # *****************************************************************
    print('\tSetting up spike/voltage monitors...')

    # General
    mon_spike = SpikeMonitor(excit_pop)
    mon_5ht_rate = PopulationRateMonitor(excit_pop)
    mon_lhb_rate = PopulationRateMonitor(lhb_pop)
    mon_pfc_rate = PopulationRateMonitor(pfc_pop)

    # Ca, pr monitors
    # Pick random synapses from synapse population since the inds
    # don't exactly map onto neuron numbers here.
    ind_syns = np.linspace(0, n_5ht, n_neur_samp).astype(int)

    # synaptic monitors
    mon_syn_kern = StateMonitor(syn_excit_excit,
                                'k',
                                record=ind_syns,
                                dt=mon_dt)
    mon_syn_effic = StateMonitor(syn_excit_excit,
                                 'efficacy',
                                 record=ind_syns,
                                 dt=mon_dt)

    # 5-HT cell monitors
    inds_record = np.sort((np.random.rand(n_neur_samp)*n_5ht).astype(int))

    mon_v = StateMonitor(excit_pop,
                         'v',
                         record=inds_record,
                         dt=mon_dt)
    mon_g_girk = StateMonitor(excit_pop,
                              'g_girk',
                              record=inds_record,
                              dt=mon_dt)
    mon_I_tot = StateMonitor(excit_pop,
                             'I_tot',
                             record=inds_record,
                             dt=mon_dt)

    # Gather and run
    # *******************************************************************
    print('\tGathering network objects...')
    net = b2.Network(excit_pop, poisson, lhb_pop, pfc_pop,
                     syn_lhb_drn, syn_pfc_drn, syn_excit_excit,
                     mon_spike, mon_5ht_rate, mon_lhb_rate,
                     mon_pfc_rate, mon_v,
                     mon_syn_kern, mon_syn_effic,
                     mon_g_girk, mon_I_tot)

    print('\t\tRunning network simulation...\n')
    net.run(sim_time, report='stdout', report_period=500*b2.ms)

    # Save variables
    # ******************************************************************
    data = SimpleNamespace(spike=mon_spike,
                           rate_5ht=mon_5ht_rate,
                           rate_lhb=mon_lhb_rate,
                           rate_pfc=mon_pfc_rate,
                           v=mon_v,
                           syn_k=mon_syn_kern,
                           syn_effic=mon_syn_effic,
                           g_girk=mon_g_girk,
                           I_tot=mon_I_tot,
                           pfc_included=True)
    data.params = SimpleNamespace(n_5ht=n_5ht,
                                  p_lhb_input=p_lhb_input,
                                  p_pfc_input=p_pfc_input,
                                  overlap_lhbpfc=overlap_lhbpfc)

    return data


def sim_lhb_paramspace(lhb_amplis=[0, 2, 4, 5, 6, 10, 20, 30, 40],
                       **kwargs):
    """
    Simulate DRN network with a parameter space of
    incoming LHb amplitudes. Store each LHb ampli in a dict.
    """

    data = {}

    for lhb_ampli in lhb_amplis:
        print(f'lhb_ampli={lhb_ampli}')
        data[str(lhb_ampli)] = sim(lhb_ampli=lhb_ampli,
                                   **kwargs)

    return data


def sim_lhb_paramspace_freq_and_pconn(
        lhb_amplis=[0, 2, 4, 5, 6, 10, 20, 30, 40],
        lhb_pconns=np.arange(0.1, 0.51, 0.1),
        **kwargs):
    """
    Simulate DRN network over the dual parameter spaces of
    fractions of 5-HT neurons innervated by LHb (see p_lhb_input
    for more detail); and LHb input frequencies

    Return the following dict: data[lhb_pconn][lhb_ampli].
    """

    lhb_pconns = np.round(lhb_pconns, decimals=2)

    data = {}

    for lhb_pconn in lhb_pconns:
        print(f'{lhb_pconn=}')
        data[str(lhb_pconn)] = {}
        for lhb_ampli in lhb_amplis:
            print(f'\t{lhb_ampli=}')
            data[str(lhb_pconn)][str(lhb_ampli)] = sim(
                lhb_ampli=lhb_ampli,
                p_lhb_input=lhb_pconn,
                **kwargs)

    return data


def sim_lhbpfc_paramspace(lhb_amplis=[0, 5, 20],
                          pfc_amplis=[0, 5, 10, 15, 20, 30, 40],
                          **kwargs):
    """
    Simulate DRN network over the dual parameter spaces of
    LHb input frequencies, and PFC input frequencies.

    Return the following dict: data[lhb_ampli][pfc_ampli]
    """

    data = {}

    for lhb_ampli in lhb_amplis:
        print(f'lhb_ampli={lhb_ampli}')
        data[str(lhb_ampli)] = {}
        for pfc_ampli in pfc_amplis:
            print(f'\tpfc_ampli={pfc_ampli}')
            data[str(lhb_ampli)][str(pfc_ampli)] = sim_lhbpfc(
                lhb_ampli=lhb_ampli, pfc_ampli=pfc_ampli, **kwargs)

    return data


def sim_lhbpfc_paramspace_pconn(
        lhb_amplis=[0, 5, 20],
        pfc_amplis=[0, 5, 10, 20, 40],
        p_conn_ee_stds=[625, 1250, 2500, 5000],
        **kwargs):
    """
    Simulate DRN network over the triple parameter spaces of
    5-ht recurrent connection probabilities,
    LHb input frequencies, and PFC input frequencies.

    Return the following dict: data[p_conn_ee_std][lhb_ampli][pfc_ampli]
    """

    data = {}

    for p_conn_ee_std in p_conn_ee_stds:
        print(f'\n\n\np_conn_ee_std={p_conn_ee_std}')
        data[str(p_conn_ee_std)] = {}
        for lhb_ampli in lhb_amplis:
            print(f'\tlhb_ampli={lhb_ampli}')
            data[str(p_conn_ee_std)][str(lhb_ampli)] = {}
            for pfc_ampli in pfc_amplis:
                print(f'\t\tpfc_ampli={pfc_ampli}')
                data[str(p_conn_ee_std)][str(lhb_ampli)][str(pfc_ampli)] \
                    = sim_lhbpfc(lhb_ampli=lhb_ampli, pfc_ampli=pfc_ampli,
                                 dict_ee_connect={
                                     'p': '0.005*exp(-1*(i-j)**2/'
                                     + f'(2*{p_conn_ee_std}**2))'},
                                 **kwargs)

    return data


def sim_lhbpfc_paramspace_overlap(
        lhb_amplis=[0, 5, 20],
        pfc_amplis=[0, 5, 10, 20, 40],
        overlaps=[0, 0.2, 0.4, 0.6],
        **kwargs):
    """
    Simulate DRN network over the triple parameter spaces of
    fractional spatial overlap of LHb and PFC inputs,
    LHb input frequencies, and PFC input frequencies.

    Return the following dict: data[overlap][lhb_ampli][pfc_ampli]
    """

    data = {}

    for overlap in overlaps:
        print(f'\n\n\noverlap={overlap}')
        data[str(overlap)] = {}
        for lhb_ampli in lhb_amplis:
            print(f'\tlhb_ampli={lhb_ampli}')
            data[str(overlap)][str(lhb_ampli)] = {}
            for pfc_ampli in pfc_amplis:
                print(f'\t\tpfc_ampli={pfc_ampli}')
                data[str(overlap)][str(lhb_ampli)][str(pfc_ampli)] \
                    = sim_lhbpfc(lhb_ampli=lhb_ampli, pfc_ampli=pfc_ampli,
                                 overlap_lhbpfc=overlap,
                                 **kwargs)

    return data

# main plotting functions. These take input from main simulation functions
# (see docstrings for full input.)
# --------------------------


def plot_lhb_inputoutput_stpctrl(data_exp, data_ctrl,
                                 t_smoothkern=0.5, lw=0.8,
                                 t_early_end_rel=0.2, t_post_end_rel=1,
                                 plt_key_left='5',
                                 plt_key_right='20',
                                 plt_t_start=1.5,
                                 plt_t_end=3,
                                 n_ex_neurs=5,
                                 n_ex_neurs_raster=2000,
                                 inset_xlim=[2.1, 2.8],
                                 inset_ylim=[-0.1, 5],
                                 figname='Fig6_LHb_input',
                                 ):

    """
    Takes input from sim_lhb_paramspace() and makes a plot of
    sample network activity during LHb input, as well as
    the input-output function of the network.
    """

    plt.style.use('publication_ml')
    fig = plt.figure(figsize=(5, 5), constrained_layout=True)
    spec = gridspec.GridSpec(nrows=6,
                             ncols=6,
                             figure=fig,
                             height_ratios=[0.5, 1, 2, 1, 1, 1])

    ax_girk = []
    ax_girk.append(fig.add_subplot(spec[4, 0:3]))
    ax_girk.append(fig.add_subplot(spec[4, 3:6], sharey=ax_girk[0]))

    ax_lhb_rate = []
    ax_lhb_rate.append(fig.add_subplot(spec[0, 0:3], sharex=ax_girk[0]))
    ax_lhb_rate.append(fig.add_subplot(spec[0, 3:6], sharex=ax_girk[1]))

    ax_drn_rate = []
    ax_drn_rate.append(fig.add_subplot(spec[1, 0:3], sharex=ax_girk[0]))
    ax_drn_rate.append(fig.add_subplot(spec[1, 3:6], sharex=ax_girk[1],
                                       sharey=ax_drn_rate[0]))

    ax_raster = []
    ax_raster.append(fig.add_subplot(spec[2, 0:3], sharex=ax_girk[0]))
    ax_raster.append(fig.add_subplot(spec[2, 3:6], sharex=ax_girk[1],
                                     sharey=ax_raster[0]))

    ax_v = []
    ax_v.append(fig.add_subplot(spec[3, 0:3], sharex=ax_girk[0]))
    ax_v.append(fig.add_subplot(spec[3, 3:6], sharex=ax_girk[1],
                                sharey=ax_v[0]))

    # Iterate through neurs
    for ind, datum in enumerate([data_exp[plt_key_left],
                                 data_exp[plt_key_right]]):

        n_neurs = len(datum.spike.spike_trains())

        if plt_t_end is False:
            plt_t_end = datum.v.t[-1] / b2.second
        colors = sns.diverging_palette(250, 15, s=90, l=40,
                                       n=n_ex_neurs, center="light")

        # 1. Mean activity of LHb
        # ---------------------------
        ax_lhb_rate[ind].plot(datum.rate_lhb.t / b2.second,
                              datum.rate_lhb.smooth_rate(
                                  width=t_smoothkern*b2.ms) / b2.Hz,
                              c="k", linewidth=lw)

        ax_lhb_rate[ind].set_ylim([-2, ax_lhb_rate[ind].get_ylim()[1]])

        # 2. Mean activity of DRN
        # ---------------------------
        rate_drn = datum.rate_5ht.smooth_rate(width=t_smoothkern*b2.ms) / b2.Hz
        ax_drn_rate[ind].plot(datum.rate_5ht.t / b2.second,
                              rate_drn,
                              c="k", linewidth=lw)

        ind_t_start = np.argmin(np.abs(
            (datum.rate_5ht.t/b2.second) - plt_t_start)).astype(int)
        ylim_rate_drn = np.max(rate_drn[ind_t_start:])
        ax_drn_rate[ind].set_ylim([-2, ylim_rate_drn])

        # 3. Raster plot of all neurons
        # ---------------------------
        ax_raster[ind].scatter(datum.spike.t / b2.second, datum.spike.i,
                               marker='.', c='k', s=2, lw=0)

        ax_raster[ind].set_ylim([-1, n_ex_neurs_raster+1])

        # 4. Example voltage traces
        # ---------------------------
        for neur in range(n_ex_neurs):
            ax_v[ind].plot(datum.v.t,
                           datum.v.v.T[:, neur] * 1000,
                           color=colors[neur],
                           linewidth=lw)

        # 5. g_GIRK
        # -----------------------------
        for neur in range(n_ex_neurs):
            ax_girk[ind].plot(datum.g_girk.t,
                              datum.g_girk.g_girk.T[:, neur] * 10**9,
                              color=colors[neur], linewidth=lw)

        # Set fig params
        # ------------------------
        if ind == 0:
            ax_lhb_rate[ind].set_ylabel('LHb (Hz)')
            ax_drn_rate[ind].set_ylabel('5-HT (Hz)')
            ax_raster[ind].set_ylabel('neur')
            ax_v[ind].set_ylabel('V (mV)')
            ax_girk[ind].set_ylabel('$g_{GIRK}$ (nS)')

        ax_girk[ind].set_xlabel('Time (s)')
        ax_girk[ind].set_xlim([plt_t_start, plt_t_end])
        plt.setp(ax_girk[ind].get_xticklabels(), visible=True)
        ax_girk[ind].spines['bottom'].set_visible(True)

    ax_drn_early = fig.add_subplot(spec[5, 0:2])
    ax_drn_late = fig.add_subplot(spec[5, 2:4])
    ax_drn_post = fig.add_subplot(spec[5, 4:6])

    conds_str = list(data_exp.keys())
    conds_float = [float(conds_str[ind]) for ind in range(len(conds_str))]

    n_conds = len(conds_str)

    # colors
    # --------------
    colors = sns.diverging_palette(250, 15, s=90, l=40,
                                   n=n_conds, center="light")
    color_early = sns.xkcd_rgb['grey']
    color_late = sns.xkcd_rgb['greenish teal']
    color_post = sns.xkcd_rgb['amethyst']
    color_ctrl = sns.xkcd_rgb['black']

    # Calculate various times
    # ---------------
    last_key_lhb = list(data_exp.keys())[-1]

    ind_stim_on = np.where(data_exp[last_key_lhb].rate_lhb.smooth_rate(
        width=0.5*b2.ms) > 0)[0][0]
    t_stim_on = data_exp[last_key_lhb].rate_lhb.t[ind_stim_on]

    ind_stim_off = np.where(data_exp[last_key_lhb].rate_lhb.smooth_rate(
        width=0.5*b2.ms) > 0)[0][-1]
    t_stim_off = data_exp[last_key_lhb].rate_lhb.t[ind_stim_off]

    # Setup early, late, post times and data_exp structures
    # -----------
    t_early_start = t_stim_on
    ind_early_start = ind_stim_on
    t_early_end = t_stim_on + t_early_end_rel*b2.second
    ind_early_end = np.argmin(np.abs(
        data_exp[last_key_lhb].rate_lhb.t - t_early_end))

    rate_early = np.empty(n_conds)
    rate_early_ctrl = np.empty(n_conds)

    # -------------
    t_late_start = t_early_end
    ind_late_start = ind_early_end
    t_late_end = t_stim_off
    ind_late_end = ind_stim_off

    rate_late = np.empty_like(rate_early)
    rate_late_ctrl = np.empty_like(rate_early_ctrl)

    # -------------
    t_post_start = t_stim_off
    ind_post_start = ind_stim_off
    t_post_end = t_post_start + t_post_end_rel*b2.second
    ind_post_end = np.argmin(np.abs(
        data_exp[last_key_lhb].rate_lhb.t - t_post_end))

    rate_post = np.empty_like(rate_early)
    rate_post_ctrl = np.empty_like(rate_early_ctrl)
    # chg_post_mean = np.empty_like(chg_early_mean)

    for ind_lhb, key_lhb in enumerate(data_exp.keys()):
        _datum = data_exp[key_lhb]
        _datum_ctrl = data_ctrl[key_lhb]

        # 1 Mean activity of DRN
        # ---------------------------
        rate_drn = _datum.rate_5ht.smooth_rate(
            width=t_smoothkern*b2.ms) / b2.Hz
        rate_drn_ctrl = _datum_ctrl.rate_5ht.smooth_rate(
            width=t_smoothkern*b2.ms) / b2.Hz

        # 2 DRN Activity measurements
        # ---------------------------
        rate_early[ind_lhb] = np.max(
            rate_drn[ind_early_start:ind_early_end])
        rate_early_ctrl[ind_lhb] = np.max(
            rate_drn_ctrl[ind_early_start:ind_early_end])

        rate_late[ind_lhb] = np.mean(
            rate_drn[ind_late_start:ind_late_end])
        rate_late_ctrl[ind_lhb] = np.mean(
            rate_drn_ctrl[ind_late_start:ind_late_end])

        rate_post[ind_lhb] = np.mean(
            rate_drn[ind_post_start:ind_post_end])
        rate_post_ctrl[ind_lhb] = np.mean(
            rate_drn_ctrl[ind_post_start:ind_post_end])


    # plot DRN activity averages
    # --------------------------------
    ax_drn_early.plot(conds_float, rate_early,
                      color=sns.xkcd_rgb['black'])
    ax_drn_early.plot(conds_float, rate_early_ctrl,
                      color=sns.xkcd_rgb['grey'])
    ax_drn_early.set_xlabel('LHb input (Hz)')
    ax_drn_early.set_ylabel('max. output (Hz)')
    ax_drn_early.set_ylim([0, ax_drn_early.get_ylim()[1]])

    ax_drn_late.plot(conds_float, rate_late,
                     color=sns.xkcd_rgb['black'])
    ax_drn_late.plot(conds_float, rate_late_ctrl,
                     color=sns.xkcd_rgb['grey'])
    ax_drn_late.set_xlabel('LHb input (Hz)')
    ax_drn_late.set_ylabel('mean output (Hz)')
    ax_drn_late.set_ylim([0, ax_drn_late.get_ylim()[1]])

    ax_drn_post.plot(conds_float, rate_post,
                     color=sns.xkcd_rgb['black'])                     
    ax_drn_post.plot(conds_float, rate_post_ctrl,
                     color=sns.xkcd_rgb['grey'])
    ax_drn_post.set_xlabel('LHb input (Hz)')
    ax_drn_post.set_ylabel('mean output (Hz)')
    ax_drn_post.set_ylim([0, ax_drn_post.get_ylim()[1]])

    fig.savefig(os.path.join('figs', f'{figname}.pdf'))
    plt.show()

    return


def plot_lhb_paramspace_freq_and_pconn(data, t_smoothkern=0.5, lw=0.8,
                                       plt_key_lhb_pconn='0.1',
                                       t_early_end_rel=0.2, t_post_end_rel=1,
                                       inset_xlim=[2.1, 2.8],
                                       inset_ylim=[-0.1, 5],
                                       figname='FigE5.pdf'):
    """
    Takes input from sim_lhb_paramspace_freq_and_pconn().
    Makes a set of plots of the input-output function of the network
    as the fraction of 5-ht neurons innervated by long-range LHb
    input is varied systematically.
    """

    n_colors = len(data.keys())

    colors_early = sns.cubehelix_palette(start=2, rot=0,
                                         dark=0.4, light=0.8,
                                         hue=0,
                                         n_colors=n_colors)
    colors_late = sns.cubehelix_palette(start=2, rot=0,
                                        dark=0.4, light=0.8,
                                        n_colors=n_colors)
    colors_post = sns.cubehelix_palette(start=0, rot=0,
                                        dark=0.5, light=0.8,
                                        n_colors=n_colors)

    plt.style.use('publication_ml')
    fig = plt.figure(figsize=(5, 1.7), constrained_layout=True)
    spec = gridspec.GridSpec(nrows=1,
                             ncols=3,
                             figure=fig)

    ax_drn_early = fig.add_subplot(spec[0, 0])
    ax_drn_late = fig.add_subplot(spec[0, 1])
    ax_drn_post = fig.add_subplot(spec[0, 2])

    conds_lhb_pconn = list(data.keys())
    conds_lhb_in = list(data[conds_lhb_pconn[0]].keys())
    conds_lhb_in_float = [float(conds_lhb_in[ind])
                          for ind in range(len(conds_lhb_in))]
    conds_lhb_pconn_float = [float(conds_lhb_pconn[ind])
                             for ind in range(len(conds_lhb_pconn))]

    n_conds_lhb_in = len(conds_lhb_in)

    # colors
    # --------------
    colors = sns.diverging_palette(250, 15, s=90, l=40,
                                   n=n_conds_lhb_in, center="light")
    color_early = sns.xkcd_rgb['grey']
    color_late = sns.xkcd_rgb['greenish teal']
    color_post = sns.xkcd_rgb['amethyst']

    # Calculate various times
    # ---------------
    last_key_lhb = conds_lhb_in[-1]

    _ex_lhb_rate = data[conds_lhb_pconn[0]][
        last_key_lhb].rate_lhb

    ind_stim_on = np.where(_ex_lhb_rate.smooth_rate(
        width=0.5*b2.ms) > 0)[0][0]
    ind_stim_off = np.where(_ex_lhb_rate.smooth_rate(
        width=0.5*b2.ms) > 0)[0][-1]

    t_stim_on = _ex_lhb_rate.t[ind_stim_on]
    t_stim_off = _ex_lhb_rate.t[ind_stim_off]

    # Setup early, late, post times and data structures
    # chg = {'early': {}, 'late': {}, 'post': {}}
    # -----------
    t_early_start = t_stim_on
    ind_early_start = ind_stim_on
    t_early_end = t_stim_on + t_early_end_rel*b2.second
    ind_early_end = np.argmin(np.abs(
        data[conds_lhb_pconn[0]][last_key_lhb].rate_lhb.t - t_early_end))
    rate_early = np.empty((len(conds_lhb_pconn), len(conds_lhb_in)))

    # -------------
    t_late_start = t_early_end
    ind_late_start = ind_early_end
    t_late_end = t_stim_off
    ind_late_end = ind_stim_off
    rate_late = np.empty_like(rate_early)

    # -------------
    t_post_start = t_stim_off
    ind_post_start = ind_stim_off
    t_post_end = t_post_start + t_post_end_rel*b2.second
    ind_post_end = np.argmin(np.abs(
        data[conds_lhb_pconn[0]][last_key_lhb].rate_lhb.t - t_post_end))
    rate_post = np.empty_like(rate_early)

    for ind_lhb_pconn, key_lhb_pconn in enumerate(conds_lhb_pconn):
        for ind_lhb_in, key_lhb_in in enumerate(conds_lhb_in):
            _datum = data[key_lhb_pconn][key_lhb_in]

            rate_drn = _datum.rate_5ht.smooth_rate(
                width=t_smoothkern*b2.ms) / b2.Hz

            # 2 DRN Activity measurements
            # ---------------------------
            rate_early[ind_lhb_pconn, ind_lhb_in] = np.max(
                rate_drn[ind_early_start:ind_early_end])
            rate_late[ind_lhb_pconn, ind_lhb_in] = np.mean(
                rate_drn[ind_late_start:ind_late_end])
            rate_post[ind_lhb_pconn, ind_lhb_in] = np.mean(
                rate_drn[ind_post_start:ind_post_end])

        # plot DRN activity averages
        # --------------------------------
        ax_drn_early.plot(conds_lhb_in_float,
                          rate_early[ind_lhb_pconn, :],
                          color=colors_early[ind_lhb_pconn])
        ax_drn_early.set_xlabel('LHb input (Hz)')
        ax_drn_early.set_ylabel('max. output (Hz)')

        ax_drn_late.plot(conds_lhb_in_float,
                         rate_late[ind_lhb_pconn, :],
                         color=colors_late[ind_lhb_pconn])
        ax_drn_late.set_xlabel('LHb input (Hz)')
        ax_drn_late.set_ylabel('mean output (Hz)')

        ax_drn_post.plot(conds_lhb_in_float,
                         rate_post[ind_lhb_pconn, :],
                         color=colors_post[ind_lhb_pconn])
        ax_drn_post.set_xlabel('LHb input (Hz)')
        ax_drn_post.set_ylabel('mean output (Hz)')

    for _ax in [ax_drn_early,
                ax_drn_late, ax_drn_post]:
        sns.despine(ax=_ax, offset={'left': 3})

    fig.savefig(os.path.join('figs', figname))
    plt.show()

    return


def plot_lhb_paramspace_freq_and_pconn_supps(data,
                                             figname='FigE5.pdf'):
    """
    Takes input from sim_lhb_paramspace_freq_and_pconn().
    Makes a plot of GIRK distribution and population averages
    as the fraction of 5-ht neurons innervated by long-range LHb
    input is varied systematically.
    """

    plt.style.use('publication_ml')
    fig = plt.figure(figsize=(3, 1.75))
    spec = gridspec.GridSpec(1, 2, figure=fig)
    ax_cdf = fig.add_subplot(spec[0, 0])
    ax_freq = fig.add_subplot(spec[0, 1])

    ax_cdf.hist(np.max(data['0.5']['20'].g_girk.g_girk.T/b2.nS, axis=0),
                bins=20, cumulative=True, histtype='step',
                color=sns.xkcd_rgb['ocean blue'],
                alpha=0.9, label='p=0.5, 20hz')
    ax_cdf.hist(np.max(data['0.1']['20'].g_girk.g_girk.T/b2.nS, axis=0),
                bins=20, cumulative=True, histtype='step',
                color=sns.xkcd_rgb['ocean blue'],
                alpha=0.4, label='p=0.1, 20hz')
    ax_cdf.hist(np.max(data['0.5']['5'].g_girk.g_girk.T/b2.nS, axis=0),
                bins=20, cumulative=True, histtype='step',
                color=sns.xkcd_rgb['coral'],
                alpha=0.9, label='p=0.5, 5hz')
    ax_cdf.hist(np.max(data['0.1']['5'].g_girk.g_girk.T/b2.nS, axis=0),
                bins=20, cumulative=True, histtype='step',
                color=sns.xkcd_rgb['coral'],
                alpha=0.4, label='p=0.1, 5hz')
    ax_cdf.legend()
    ax_cdf.set_xlabel('$g_{GIRK}$ (nS)')
    ax_cdf.set_ylabel('count')

    g_girk_mean = {}

    colors = sns.cubehelix_palette(start=2, rot=0,
                                   dark=0.4, light=0.8,
                                   n_colors=len(data.keys()))
    for ind, cond in enumerate(list(data.keys())):
        g_girk_mean[cond] = np.zeros(len(data[cond].keys()))
        for ind_lhbin, cond_lhbin in enumerate(list(data[cond].keys())):
            g_girk_mean[cond][ind_lhbin] = np.mean(np.max(
                data[cond][cond_lhbin].g_girk.g_girk.T/b2.nS, axis=0))

        ax_freq.plot(np.array(list(data[cond].keys())).astype(np.int),
                     g_girk_mean[cond][:], color=colors[ind],
                     label=f'p={cond}')

    ax_freq.legend()
    ax_freq.set_xlabel('freq')
    ax_freq.set_ylabel('$g_{GIRK}$ (nS) (pop. mean)')

    fig.savefig(os.path.join('figs', figname))
    plt.show()

    return


def plot_lhbpfc_paramspace(data,
                           plt_lhb_key_high='20',
                           plt_lhb_key_low='5',
                           plt_pfc_key='10',
                           t_smoothkern=5,
                           pfc_start=2, pfc_end=3,
                           lhb_start=2.5, lhb_end=3,
                           colorpal_lhb=sns.color_palette('Blues',
                                                          n_colors=3),
                           color_pfc=sns.xkcd_rgb['grass green'],
                           color_5ht_lhbin=sns.xkcd_rgb['purple'],
                           color_5ht_pfcin=sns.xkcd_rgb['orange'],
                           color_5ht_lhbpfcin=sns.xkcd_rgb['burnt sienna'],
                           figname='plt_lhbpfc_paramspace.pdf'):

    """
    Takes input from sim_lhbpfc_paramspace().
    Makes a plot of example DRN simulations and winner-take-all effects
    of input-output transformation.
    """

    plt.style.use('publication_ml')
    fig = plt.figure(figsize=(3.35, 2), constrained_layout=True)
    spec = gridspec.GridSpec(nrows=4,
                             ncols=3,
                             figure=fig,
                             height_ratios=[0.3, 0.3, 0.3, 1],
                             width_ratios=[1, 1, 0.85])

    ax_compar_mean = fig.add_subplot(spec[3, 2])

    ax_lowlhb_raster = fig.add_subplot(spec[3, 0])
    ax_highlhb_raster = fig.add_subplot(spec[3, 1],
                                        sharey=ax_lowlhb_raster)

    ax_lowlhb_lhb_rate = fig.add_subplot(spec[1, 0],
                                         sharex=ax_lowlhb_raster)
    ax_highlhb_lhb_rate = fig.add_subplot(spec[1, 1],
                                          sharex=ax_highlhb_raster,
                                          sharey=ax_lowlhb_lhb_rate)

    ax_lowlhb_pfc_rate = fig.add_subplot(spec[0, 0],
                                         sharex=ax_lowlhb_raster)
    ax_highlhb_pfc_rate = fig.add_subplot(spec[0, 1],
                                          sharex=ax_highlhb_raster,
                                          sharey=ax_lowlhb_pfc_rate)

    ax_lowlhb_5htrate = fig.add_subplot(spec[2, 0],
                                        sharex=ax_lowlhb_raster)
    ax_highlhb_5htrate = fig.add_subplot(spec[2, 1],
                                         sharex=ax_highlhb_raster,
                                         sharey=ax_lowlhb_5htrate)

    # Setup data
    # ----------------------
    # Setup lhb keys
    keys_lhb = list(data.keys())
    for ind, key in enumerate(keys_lhb):
        if key == plt_lhb_key_low:
            ind_plt_lhb_key_low = ind
        elif key == plt_lhb_key_high:
            ind_plt_lhb_key_high = ind

    _datum = {}
    _datum['l'] = data[plt_lhb_key_low][plt_pfc_key]
    _datum['h'] = data[plt_lhb_key_high][plt_pfc_key]

    n_neurs = max(_datum['l'].spike.i)
    ind_low_lhb = 0
    ind_high_lhb = ind_low_lhb + int(n_neurs * _datum['l'].params.p_lhb_input)
    ind_low_pfc = int(ind_high_lhb
                      - (_datum['l'].params.overlap_lhbpfc
                         * _datum['l'].params.p_lhb_input * n_neurs))
    ind_high_pfc = ind_low_pfc + int(n_neurs * _datum['l'].params.p_pfc_input)

    # 0.0 Mean activity of LHb
    # ---------------------------
    t_vector = _datum['l'].rate_lhb.t
    rate_lhb = {}
    rate_pfc = {}

    for key in ['l', 'h']:
        rate_lhb[key] = SimpleNamespace(t=t_vector, rate=np.zeros_like(t_vector))
        rate_pfc[key] = SimpleNamespace(t=t_vector, rate=np.zeros_like(t_vector))

    ax_lowlhb_lhb_rate.plot(_datum['l'].rate_lhb.t / b2.second,
                            _datum['l'].rate_lhb.smooth_rate(
                                width=t_smoothkern*b2.ms) / b2.Hz,
                            color=colorpal_lhb[ind_plt_lhb_key_low],
                            linewidth=0.5)
    ax_highlhb_lhb_rate.plot(_datum['h'].rate_lhb.t / b2.second,
                             _datum['h'].rate_lhb.smooth_rate(
                                 width=t_smoothkern*b2.ms) / b2.Hz,
                             color=colorpal_lhb[ind_plt_lhb_key_high],
                             linewidth=0.5)

    ax_lowlhb_lhb_rate.set_ylabel('LHb\n(Hz)')
    ax_lowlhb_lhb_rate.set_ylim([-2, ax_highlhb_lhb_rate.get_ylim()[1]])
    sns.despine(ax=ax_lowlhb_lhb_rate, bottom=True)
    sns.despine(ax=ax_highlhb_lhb_rate, bottom=True)

    # 0.1 Mean activity of PFC
    # ---------------------------
    ax_lowlhb_pfc_rate.plot(_datum['l'].rate_pfc.t / b2.second,
                            _datum['l'].rate_pfc.smooth_rate(
                                width=t_smoothkern*b2.ms) / b2.Hz,
                            c=color_pfc, linewidth=0.5)
    ax_highlhb_pfc_rate.plot(_datum['h'].rate_pfc.t / b2.second,
                             _datum['h'].rate_pfc.smooth_rate(
                                 width=t_smoothkern*b2.ms) / b2.Hz,
                             c=color_pfc, linewidth=0.5)

    ax_lowlhb_pfc_rate.set_ylabel('InA\n(Hz)')
    ax_highlhb_pfc_rate.set_ylim([-2, ax_highlhb_pfc_rate.get_ylim()[1]])
    ax_lowlhb_pfc_rate.set_ylim([-2, ax_highlhb_pfc_rate.get_ylim()[1]])
    sns.despine(ax=ax_lowlhb_pfc_rate, bottom=True)
    sns.despine(ax=ax_highlhb_pfc_rate, bottom=True)

    # 1. Raster plot and 5-HT rates
    # ---------------------------
    _axs_raster = [ax_lowlhb_raster, ax_highlhb_raster]
    _axs_rate = [ax_lowlhb_5htrate, ax_highlhb_5htrate]
    inds_raster_lhbrecip = {}
    inds_raster_pfcrecip = {}
    inds_raster_lhbpfcrecip = {}
    inds_raster_therest = {}

    for ind, key in enumerate(['l', 'h']):
        inds_raster_lhbrecip[key] = np.logical_and(
            _datum[key].spike.i > ind_low_lhb,
            _datum[key].spike.i < ind_high_lhb)
        inds_raster_pfcrecip[key] = np.logical_and(
            _datum[key].spike.i > ind_low_pfc,
            _datum[key].spike.i < ind_high_pfc)
        inds_raster_lhbpfcrecip[key] = np.logical_and(
            _datum[key].spike.i < ind_high_lhb,
            _datum[key].spike.i > ind_low_pfc)
        inds_raster_therest[key] = _datum[key].spike.i > ind_high_pfc

        # Plot the rasters
        _axs_raster[ind].scatter(
            _datum[key].spike.t[inds_raster_lhbrecip[key]] / b2.second,
            _datum[key].spike.i[inds_raster_lhbrecip[key]],
            marker='.', color=color_5ht_lhbin, s=2, lw=0)
        _axs_raster[ind].scatter(
            _datum[key].spike.t[inds_raster_pfcrecip[key]] / b2.second,
            _datum[key].spike.i[inds_raster_pfcrecip[key]],
            marker='.', color=color_5ht_pfcin, s=2, lw=0)
        _axs_raster[ind].scatter(
            _datum[key].spike.t[inds_raster_lhbpfcrecip[key]] / b2.second,
            _datum[key].spike.i[inds_raster_lhbpfcrecip[key]],
            marker='.', color=color_5ht_lhbpfcin, s=2, lw=0)   
        _axs_raster[ind].scatter(
            _datum[key].spike.t[inds_raster_therest[key]] / b2.second,
            _datum[key].spike.i[inds_raster_therest[key]],
            marker='.', color='k', s=2, lw=0)

        _axs_raster[ind].set_ylim([-1, ind_high_pfc+1])

        # Plot the rates
        _fr = calc_fr_from_spikemonitor(
            _datum[key].spike,
            ind_neur_min=ind_low_pfc,
            ind_neur_max=ind_high_pfc,
            dt=t_smoothkern*2*b2.ms)

        _axs_rate[ind].plot(_fr.t, _fr.rate, color=color_5ht_pfcin,
                            linewidth=0.5)
        sns.despine(ax=_axs_rate[ind], bottom=True)

    ax_lowlhb_raster.set_ylabel('5-HT neuron')
    ax_lowlhb_raster.set_xlabel('Time (s)')
    ax_highlhb_raster.set_xlabel('Time (s)')

    ax_lowlhb_5htrate.set_xlim([1.8, 3.2])
    ax_highlhb_5htrate.set_xlim([1.8, 3.2])

    ax_lowlhb_5htrate.set_ylim(-2, ax_lowlhb_5htrate.get_ylim()[1])
    ax_lowlhb_5htrate.set_ylabel('A-recip.\n5-HT (Hz)')


    # 2 Strip axes of unneccesary tickmarks
    # -----------
    plt.setp(ax_highlhb_lhb_rate.get_yticklabels(), visible=False)
    plt.setp(ax_highlhb_pfc_rate.get_yticklabels(), visible=False)
    plt.setp(ax_highlhb_raster.get_yticklabels(), visible=False)
    plt.setp(ax_highlhb_5htrate.get_yticklabels(), visible=False)

    plt.setp(ax_highlhb_lhb_rate.get_xticklabels(), visible=False)
    plt.setp(ax_lowlhb_lhb_rate.get_xticklabels(), visible=False)
    plt.setp(ax_highlhb_pfc_rate.get_xticklabels(), visible=False)
    plt.setp(ax_lowlhb_pfc_rate.get_xticklabels(), visible=False)
    plt.setp(ax_highlhb_5htrate.get_xticklabels(), visible=False)
    plt.setp(ax_lowlhb_5htrate.get_xticklabels(), visible=False)

    # 3. Mean response of PFC-recipient neurons
    # ---------------------------
    keys_lhb = list(data.keys())
    keys_pfc = list(data[keys_lhb[0]].keys())

    list_pfc_conds = [int(key) for key in keys_pfc]

    rate_lhb = {'mean': {}, 'max': {}}
    rate_pfc = {'mean': {}, 'max': {}}

    for ind, key_lhb in enumerate(keys_lhb):

        for _rate in [rate_lhb, rate_pfc]:
            for _analysis_key in ['mean', 'max']:
                _rate[_analysis_key][key_lhb] = {}

        for key_pfc in keys_pfc:
            _data = data[key_lhb][key_pfc]

            _rates_pfc = calc_neur_fr(
                _data,
                ind_neurs=np.arange(ind_low_pfc, ind_high_pfc),
                t_start=lhb_start, t_end=lhb_end)
            _mean_pfc = np.mean(_rates_pfc)
            rate_pfc['mean'][key_lhb][key_pfc] = _mean_pfc

        ax_compar_mean.plot(list_pfc_conds,
                            list(rate_pfc['mean'][key_lhb].values()),
                            color=colorpal_lhb[ind],
                            label=f'LHb={key_lhb}',
                            linewidth=0.5)

    ax_compar_mean.set_xlim([0, 40])
    ax_compar_mean.legend()
    ax_compar_mean.set_xlabel('PFC input (Hz)')
    ax_compar_mean.set_ylabel('mean DRN\noutput (Hz)')

    fig.savefig(os.path.join('figs', figname))
    plt.show()

    return


def plot_lhbpfc_paramspace_pconn(data,
                                 lhb_start=2.5, lhb_end=3,
                                 n_neurs=5000, p_lhb=0.25,
                                 p_pfc=0.25, plt_ylim=[-5, 0],
                                 figname='FigE6.pdf'):
    """
    Takes input from sim_lhbpfc_paramspace_pconn().
    Makes a plot quantifying the winner-take-all effect of DRN
    network during simultaneous LHB and Area A input,
    as a function of the spatial scale of 5-HT recurrent connectivity.
    """
    ind_low_lhb = 0
    ind_high_lhb = ind_low_lhb + int(n_neurs * p_lhb)
    ind_low_pfc = ind_high_lhb
    ind_high_pfc = ind_low_pfc + int(n_neurs * p_pfc)

    keys_pconn = list(data.keys())
    keys_lhb = list(data[keys_pconn[0]].keys())
    keys_pfc = list(data[keys_pconn[0]][keys_lhb[0]].keys())

    list_pfc_conds = [int(key) for key in keys_pfc]

    rate_lhb = {'mean': {}, 'max': {}}
    rate_pfc = {'mean': {}, 'max': {}}

    for ind_pconn, key_pconn in enumerate(keys_pconn):
        for _rate in [rate_lhb, rate_pfc]:
            for _analysis_key in ['mean', 'max']:
                _rate[_analysis_key][key_pconn] = {}

        for ind_lhb, key_lhb in enumerate(keys_lhb):

            for _rate in [rate_lhb, rate_pfc]:
                for _analysis_key in ['mean', 'max']:
                    _rate[_analysis_key][key_pconn][key_lhb] = {}

            for key_pfc in keys_pfc:
                _data = data[key_pconn][key_lhb][key_pfc]

                _rates_pfc = calc_neur_fr(
                    _data,
                    ind_neurs=np.arange(ind_low_pfc, ind_high_pfc),
                    t_start=lhb_start, t_end=lhb_end)
                _mean_pfc = np.mean(_rates_pfc)
                rate_pfc['mean'][key_pconn][key_lhb][key_pfc] = _mean_pfc

    # calculate modulation index
    mod_index = {}
    for ind_pconn, key_pconn in enumerate(keys_pconn):
        mod_index[key_pconn] = {'20vs0': np.zeros(len(keys_pfc)),
                                '5vs0': np.zeros(len(keys_pfc))}
        for ind_pfc, key_pfc in enumerate(keys_pfc):

            _mod_index = (rate_pfc['mean'][key_pconn]['20'][key_pfc]
                          - rate_pfc['mean'][key_pconn]['0'][key_pfc])
            mod_index[key_pconn]['20vs0'][ind_pfc] = _mod_index

            _mod_index = (rate_pfc['mean'][key_pconn]['5'][key_pfc]
                          - rate_pfc['mean'][key_pconn]['0'][key_pfc])
            mod_index[key_pconn]['5vs0'][ind_pfc] = _mod_index

    # -------------
    plt.style.use('publication_ml')
    fig = plt.figure(figsize=(3, 1.75))
    spec = gridspec.GridSpec(1, 2, figure=fig)
    ax_5vs0 = fig.add_subplot(spec[0, 0])
    ax_20vs0 = fig.add_subplot(spec[0, 1], sharey=ax_5vs0)

    colors = sns.cubehelix_palette(start=2, rot=0,
                                   dark=0.4, light=0.8,
                                   n_colors=len(data.keys()))

    for ind_pconn, key_pconn in enumerate(keys_pconn):
        ax_20vs0.plot(np.array(keys_pfc).astype(np.int),
                      mod_index[key_pconn]['20vs0'], color=colors[ind_pconn],
                      label=f'pconn={key_pconn}')
        ax_5vs0.plot(np.array(keys_pfc).astype(np.int),
                     mod_index[key_pconn]['5vs0'], color=colors[ind_pconn],
                     label=f'pconn={key_pconn}')

    ax_20vs0.legend()
    ax_5vs0.legend()

    ax_20vs0.plot(np.array(keys_pfc).astype(np.int),
                  np.zeros_like(np.array(keys_pfc).astype(np.int)),
                  '--', color='k', linewidth=0.5)
    ax_5vs0.plot(np.array(keys_pfc).astype(np.int),
                 np.zeros_like(np.array(keys_pfc).astype(np.int)),
                 '--', color='k', linewidth=0.5)

    ax_20vs0.set_ylabel('LHb-driven $\Delta \mathregular{{5-HT}_{A-recip.}}$ resp. (Hz)')
    ax_5vs0.set_ylabel('LHb-driven $\Delta \mathregular{{5-HT}_{A-recip.}}$ resp. (Hz)')
    ax_20vs0.set_title('20Hz LHb input')
    ax_5vs0.set_title('5Hz LHb input')
    ax_20vs0.set_xlabel('PFC input (Hz)')
    ax_5vs0.set_xlabel('PFC input (Hz)')

    ax_20vs0.set_ylim(plt_ylim)
    ax_5vs0.set_ylim(plt_ylim)

    fig.savefig(os.path.join('figs', figname))
    plt.show()

    return


def plot_lhbpfc_paramspace_overlap(data,
                                   lhb_start=2.5, lhb_end=3,
                                   plt_ylim=None,
                                   figname='FigE8.pdf'):
    """
    Takes input from sim_lhbpfc_paramspace_overlap().
    Makes a plot quantifying the winner-take-all effect of DRN
    network during simultaneous LHB and Area A input,
    as a function of the overlap of LHb and A input onto
    5-HT populations.
    """

    # Setup data
    # ----------------------
    # Setup lhb keys
    keys_overlap = list(data.keys())
    keys_lhb = list(data[keys_overlap[0]].keys())
    keys_pfc = list(data[keys_overlap[0]][keys_lhb[0]].keys())


    # -------------------
    rate_lhb = {'transient': {}, 'stationary': {}, 'all': {}}
    rate_pfc = {'transient': {}, 'stationary': {}, 'all': {}}

    t_transient_bounds = [0, 0.2]
    t_stationary_bounds = [0.2, 0.5]
    t_all_bounds = [0, 0.5]

    # setup structures to store rate information
    for _rate in [rate_lhb, rate_pfc]:
        for _analysis_key in ['transient', 'stationary', 'all']:
            for ind_overlap, key_overlap in enumerate(keys_overlap):
                _rate[_analysis_key][key_overlap] = {}
                for ind_lhb, key_lhb in enumerate(keys_lhb):
                    _rate[_analysis_key][key_overlap][key_lhb] = {}

    # store rate information
    for ind_overlap, key_overlap in enumerate(keys_overlap):
        for ind_lhb, key_lhb in enumerate(keys_lhb):
            for ind_pfc, key_pfc in enumerate(keys_pfc):
                # calc indices of cells for this sim
                _data = data[key_overlap][key_lhb][key_pfc]

                n_neurs = max(_data.spike.i)
                ind_low_lhb = 0
                ind_high_lhb = ind_low_lhb + int(
                    n_neurs * _data.params.p_lhb_input)
                ind_low_pfc = int(ind_high_lhb
                                  - (_data.params.overlap_lhbpfc
                                     * _data.params.p_lhb_input * n_neurs))
                ind_high_pfc = ind_low_pfc + int(
                    n_neurs * _data.params.p_pfc_input)

                # calc firing rates
                _rates_pfc_transient = calc_neur_fr(
                    _data,
                    ind_neurs=np.arange(ind_low_pfc, ind_high_pfc),
                    t_start=lhb_start+t_transient_bounds[0],
                    t_end=lhb_start+t_transient_bounds[1])
                _mean_pfc_transient = np.mean(_rates_pfc_transient)
                rate_pfc['transient'][key_overlap][key_lhb][key_pfc] \
                    = _mean_pfc_transient

                _rates_pfc_stationary = calc_neur_fr(
                    _data,
                    ind_neurs=np.arange(ind_low_pfc, ind_high_pfc),
                    t_start=lhb_start+t_stationary_bounds[0],
                    t_end=lhb_start+t_stationary_bounds[1])
                _mean_pfc_stationary = np.mean(_rates_pfc_stationary)
                rate_pfc['stationary'][key_overlap][key_lhb][key_pfc] \
                    = _mean_pfc_stationary

                _rates_pfc_all = calc_neur_fr(
                    _data,
                    ind_neurs=np.arange(ind_low_pfc, ind_high_pfc),
                    t_start=lhb_start+t_all_bounds[0],
                    t_end=lhb_start+t_all_bounds[1])
                _mean_pfc_all = np.mean(_rates_pfc_all)
                rate_pfc['all'][key_overlap][key_lhb][key_pfc] \
                    = _mean_pfc_all

    # calculate modulation index
    mod_index = {'stationary': {}, 'transient': {}, 'all': {}}

    for resp_phase in ['stationary', 'transient', 'all']:
        for ind_overlap, key_overlap in enumerate(keys_overlap):
            mod_index[resp_phase][key_overlap] = {
                '20vs0': np.zeros(len(keys_pfc)),
                '5vs0': np.zeros(len(keys_pfc))}

            for ind_pfc, key_pfc in enumerate(keys_pfc):
                _mod_index_20vs0 = (
                    rate_pfc[resp_phase][key_overlap]['20'][key_pfc]
                    - rate_pfc[resp_phase][key_overlap]['0'][key_pfc])
                mod_index[resp_phase][key_overlap]['20vs0'][ind_pfc] \
                    = _mod_index_20vs0

                _mod_index_5vs0 = (
                    rate_pfc[resp_phase][key_overlap]['5'][key_pfc]
                    - rate_pfc[resp_phase][key_overlap]['0'][key_pfc])
                mod_index[resp_phase][key_overlap]['5vs0'][ind_pfc] \
                    = _mod_index_5vs0

    # -------------
    plt.style.use('publication_ml')
    fig = plt.figure(figsize=(3, 4.5))
    spec = gridspec.GridSpec(3, 2, figure=fig)
    ax_5vs0_t = fig.add_subplot(spec[0, 0])
    ax_20vs0_t = fig.add_subplot(spec[0, 1], sharey=ax_5vs0_t)

    ax_5vs0_s = fig.add_subplot(spec[1, 0])
    ax_20vs0_s = fig.add_subplot(spec[1, 1], sharey=ax_5vs0_s)

    ax_5vs0_a = fig.add_subplot(spec[2, 0])
    ax_20vs0_a = fig.add_subplot(spec[2, 1], sharey=ax_5vs0_a)

    colors = sns.cubehelix_palette(start=2, rot=0,
                                   dark=0.4, light=0.8,
                                   n_colors=len(data.keys()))

    for ind_overlap, key_overlap in enumerate(keys_overlap):
        ax_20vs0_t.plot(np.array(keys_pfc).astype(np.int),
                        mod_index['transient'][key_overlap]['20vs0'],
                        color=colors[ind_overlap],
                        label=f'overlap={key_overlap}')
        ax_5vs0_t.plot(np.array(keys_pfc).astype(np.int),
                       mod_index['transient'][key_overlap]['5vs0'],
                       color=colors[ind_overlap],
                       label=f'overlap={key_overlap}')

        ax_20vs0_s.plot(np.array(keys_pfc).astype(np.int),
                        mod_index['stationary'][key_overlap]['20vs0'],
                        color=colors[ind_overlap],
                        label=f'overlap={key_overlap}')
        ax_5vs0_s.plot(np.array(keys_pfc).astype(np.int),
                       mod_index['stationary'][key_overlap]['5vs0'],
                       color=colors[ind_overlap],
                       label=f'overlap={key_overlap}')

        ax_20vs0_a.plot(np.array(keys_pfc).astype(np.int),
                        mod_index['all'][key_overlap]['20vs0'],
                        color=colors[ind_overlap],
                        label=f'overlap={key_overlap}')
        ax_5vs0_a.plot(np.array(keys_pfc).astype(np.int),
                       mod_index['all'][key_overlap]['5vs0'],
                       color=colors[ind_overlap],
                       label=f'overlap={key_overlap}')

    for _ax in [ax_20vs0_s, ax_5vs0_s, ax_20vs0_t, ax_5vs0_t,
                ax_20vs0_a, ax_5vs0_a]:
        _ax.legend()
        _ax.plot(np.array(keys_pfc).astype(np.int),
                 np.zeros_like(np.array(keys_pfc).astype(np.int)),
                 '--', color='k', linewidth=0.5)
        _ax.set_xlabel('PFC input (Hz)')
        _ax.set_ylabel('$\Delta$ A-recip. 5-HT, LHb on vs off (Hz)')
        if plt_ylim is not None:
            _ax.set_ylim(plt_ylim)

    ax_5vs0_t.set_title('transient')
    ax_5vs0_s.set_title('stationary')
    ax_5vs0_a.set_title('all')

    fig.savefig(os.path.join('figs', figname))
    plt.show()

    return


def plot_lhbpfc_paramspace_wta_quantification(data,
                                              lhb_start=2.5, lhb_end=3,
                                              mod_index_type='subtractive',
                                              plt_resp_phase='stationary',
                                              plt_cmap='coolwarm',
                                              plt_vrange=None,
                                              figname='FigE6_wta.pdf'):
    """
    Takes input from sim_lhbpfc_paramspace_**().
    (Any function starting like this will do.)
    Makes a gradient map quantifying the winner-take-all effect of DRN
    network as a function of different frequencies of LHb and A input
    """

    # Setup data
    # ----------------------
    # Setup lhb keys
    keys_lhb = list(data.keys())
    keys_pfc = list(data[keys_lhb[0]].keys())


    # -------------------
    rate_lhb = {'transient': {}, 'stationary': {}, 'all': {}}
    rate_pfc = {'transient': {}, 'stationary': {}, 'all': {}}

    t_transient_bounds = [0, 0.2]
    t_stationary_bounds = [0.2, 0.5]
    t_all_bounds = [0, 0.5]

    # setup structures to store rate information
    for _rate in [rate_lhb, rate_pfc]:
        for _analysis_key in ['transient', 'stationary', 'all']:
            for ind_lhb, key_lhb in enumerate(keys_lhb):
                _rate[_analysis_key][key_lhb] = {}

    # store rate information
    for ind_lhb, key_lhb in enumerate(keys_lhb):
        for ind_pfc, key_pfc in enumerate(keys_pfc):
            # calc indices of cells for this sim
            _data = data[key_lhb][key_pfc]

            n_neurs = max(_data.spike.i)
            ind_low_lhb = 0
            ind_high_lhb = ind_low_lhb + int(
                n_neurs * _data.params.p_lhb_input)
            ind_low_pfc = int(ind_high_lhb
                              - (_data.params.overlap_lhbpfc
                                 * _data.params.p_lhb_input * n_neurs))
            ind_high_pfc = ind_low_pfc + int(
                n_neurs * _data.params.p_pfc_input)

            # calc firing rates
            _rates_pfc_transient = calc_neur_fr(
                _data,
                ind_neurs=np.arange(ind_low_pfc, ind_high_pfc),
                t_start=lhb_start+t_transient_bounds[0],
                t_end=lhb_start+t_transient_bounds[1])
            _mean_pfc_transient = np.mean(_rates_pfc_transient)
            rate_pfc['transient'][key_lhb][key_pfc] \
                = _mean_pfc_transient

            _rates_pfc_stationary = calc_neur_fr(
                _data,
                ind_neurs=np.arange(ind_low_pfc, ind_high_pfc),
                t_start=lhb_start+t_stationary_bounds[0],
                t_end=lhb_start+t_stationary_bounds[1])
            _mean_pfc_stationary = np.mean(_rates_pfc_stationary)
            rate_pfc['stationary'][key_lhb][key_pfc] \
                = _mean_pfc_stationary

            _rates_pfc_all = calc_neur_fr(
                _data,
                ind_neurs=np.arange(ind_low_pfc, ind_high_pfc),
                t_start=lhb_start+t_all_bounds[0],
                t_end=lhb_start+t_all_bounds[1])
            _mean_pfc_all = np.mean(_rates_pfc_all)
            rate_pfc['all'][key_lhb][key_pfc] \
                = _mean_pfc_all

    # calculate modulation index
    mod_index = {'stationary': {}, 'transient': {}, 'all': {}}

    for resp_phase in ['stationary', 'transient', 'all']:
        mod_index[resp_phase] = np.zeros((len(keys_lhb), len(keys_pfc)))
        for ind_lhb, key_lhb in enumerate(keys_lhb):
            for ind_pfc, key_pfc in enumerate(keys_pfc):
                if mod_index_type == 'subtractive':
                    mod_index[resp_phase][ind_lhb, ind_pfc] = (
                        rate_pfc[resp_phase][key_lhb][key_pfc]
                        - rate_pfc[resp_phase]['0'][key_pfc])
                elif mod_index_type == 'divisive':
                    mod_index[resp_phase][ind_lhb, ind_pfc] = (
                        rate_pfc[resp_phase][key_lhb][key_pfc]
                        - rate_pfc[resp_phase]['0'][key_pfc]) \
                        / rate_pfc[resp_phase]['0'][key_pfc]

    # -------------
    plt.style.use('publication_ml')
    fig = plt.figure(figsize=(2.5, 2))
    spec = gridspec.GridSpec(1, 1, figure=fig)
    ax_img = fig.add_subplot(spec[0, 0])

    _lhb_rates_int = np.array(list(data.keys())).astype(int)
    _pfc_rates_int = np.array(list(data['0'].keys())).astype(int)

    if plt_vrange is None:
        if mod_index_type == 'subtractive':
            _vmax = np.max(np.abs(mod_index[plt_resp_phase]))
            _vmin = -1 * _vmax
        elif mod_index_type == 'divisive':
            _vmin = -1 * np.max(np.abs(mod_index[plt_resp_phase]))
            _vmax = -1 * _vmin
    elif plt_vrange is not None:
        _vmin = plt_vrange[0]
        _vmax = plt_vrange[1]

    img = ax_img.imshow(mod_index[plt_resp_phase], cmap=plt_cmap,
                        vmin=_vmin, vmax=_vmax)

    ax_img.set_xlabel('A input (Hz)')
    ax_img.set_ylabel('LHb input (Hz)')
    ax_img.set_xticks(np.arange(0, len(_pfc_rates_int)))
    ax_img.set_yticks(np.arange(0, len(_lhb_rates_int)))
    ax_img.set_xticklabels(_pfc_rates_int)
    ax_img.set_yticklabels(_lhb_rates_int)
    fig.colorbar(img, ax=ax_img)

    # fig.savefig(f'figs/fig_lhbpfc_wta_paramspace_index={mod_index_type}.pdf')
    fig.savefig(os.path.join('figs', figname))
    plt.show()

    return mod_index


# Plot main figs. This generates figure panels in Lynn et al (2025).
# See relevant simulation and plotting functions above for more info.
# ------------------


def plot_Fig6():
    """
    Plots Fig. 6 from Lynn et al. (2025).
    """

    # simulate 5-HT network under +STP and -STP conditions
    data = sim_lhb_paramspace()
    data_stp_ctrl = sim_lhb_paramspace(tau_stp_sht=1 * b2.ms,
                                       tau_stp_lng_pos=1 * b2.ms,
                                       tau_stp_lng_neg=1 * b2.ms)

    # plot simulation
    plot_lhb_inputoutput_stpctrl(data, data_stp_ctrl)

    return


def plot_Fig7():
    """
    Plots Fig. 7 from Lynn et al. (2025).
    """

    # simulate 5-HT network under +STP and -STP conditions
    data = sim_lhbpfc_paramspace()
    data_stp_ctrl = sim_lhbpfc_paramspace(tau_stp_sht=1 * b2.ms,
                                          tau_stp_lng_pos=1 * b2.ms,
                                          tau_stp_lng_neg=1 * b2.ms)

    # plot simulation
    plot_lhbpfc_paramspace(data, figname='Fig7_a-d.pdf')
    plot_lhbpfc_paramspace(data_stp_ctrl, figname='Fig7_e-g.pdf')
    return


def plot_ExtendedDataFig5():
    """
    Plots Extended Data Fig. 5 from Lynn et al. (2025).
    """

    # simulate 5-HT network under +STP and -STP conditions
    data = sim_lhb_paramspace_freq_and_pconn()
    data_stp_ctrl = sim_lhb_paramspace_freq_and_pconn(
        tau_stp_sht=1 * b2.ms,
        tau_stp_lng_pos=1 * b2.ms,
        tau_stp_lng_neg=1 * b2.ms)

    # plot simulation
    plot_lhb_paramspace_freq_and_pconn(data, figname='FigE5_a-c.pdf')
    plot_lhb_paramspace_freq_and_pconn_supps(data, figname='FigE5_d-e.pdf')

    plot_lhb_paramspace_freq_and_pconn(data_stp_ctrl, figname='FigE5_f-h.pdf')
    plot_lhb_paramspace_freq_and_pconn_supps(data_stp_ctrl,
                                             figname='FigE5_i-j.pdf')

    return


def plot_ExtendedDataFig6(lhb_amplis=[0, 5, 10, 15, 20, 25],
                          pfc_amplis=[0, 5, 10, 15, 20, 25]):
    """
    Plots Extended Data Fig. 6 from Lynn et al. (2025).

    Note that lhb_amplis and pfc_amplis can be shortened to
    decrease simulation time, as the full parameter space can take
    quite a long time to run simulations of.
    """

    # simulate 5-HT network under +STP and -STP conditions
    data = sim_lhbpfc_paramspace_pconn(lhb_amplis=lhb_amplis,
                                       pfc_amplis=pfc_amplis)
    data_stp_ctrl = sim_lhbpfc_paramspace_pconn(
        lhb_amplis=lhb_amplis,
        pfc_amplis=pfc_amplis,
        tau_stp_sht=1 * b2.ms,
        tau_stp_lng_pos=1 * b2.ms,
        tau_stp_lng_neg=1 * b2.ms)

    # plot simulation
    plot_lhbpfc_paramspace_pconn(data,
                                       figname='FigE6_d.pdf')
    plot_lhbpfc_paramspace_wta_quantification(data['1250'],
                               figname='FigE6_e_left.pdf')
    plot_lhbpfc_paramspace_wta_quantification(data['5000'],
                               figname='FigE6_e_right.pdf')

    plot_lhbpfc_paramspace_pconn(data_stp_ctrl,
                                       figname='FigE6_f.pdf')
    plot_lhbpfc_paramspace_wta_quantification(data_stp_ctrl['1250'],
                               figname='FigE6_g_left.pdf')
    plot_lhbpfc_paramspace_wta_quantification(data_stp_ctrl['5000'],
                               figname='FigE6_g_right.pdf')
    return


def plot_ExtendedDataFig7(lhb_amplis=[0, 5, 10, 15, 20, 25],
                          pfc_amplis=[0, 5, 10, 15, 20, 25]):
    """
    Plots Extended Data Fig. 7 from Lynn et al. (2025).

    Note that lhb_amplis and pfc_amplis can be shortened to
    decrease simulation time, as the full parameter space can take
    quite a long time to run simulations of.
    """

    # simulate 5-HT network under +STP and -STP conditions
    data = sim_lhbpfc_paramspace_overlap(lhb_amplis=lhb_amplis,
                                         pfc_amplis=pfc_amplis)
    data_stp_ctrl = sim_lhbpfc_paramspace_overlap(
        lhb_amplis=lhb_amplis,
        pfc_amplis=pfc_amplis,
        tau_stp_sht=1 * b2.ms,
        tau_stp_lng_pos=1 * b2.ms,
        tau_stp_lng_neg=1 * b2.ms)

    # plot simulation
    plot_lhbpfc_paramspace_overlap(data,
                                         figname='FigE7_d-e.pdf')
    plot_lhbpfc_paramspace_wta_quantification(data['0'],
                               figname='FigE7_f_left.pdf')
    plot_lhbpfc_paramspace_wta_quantification(data['0.4'],
                               figname='FigE7_f_right.pdf')

    plot_lhbpfc_paramspace_overlap(data_stp_ctrl,
                                         figname='FigE7_g-h.pdf')
    plot_lhbpfc_paramspace_wta_quantification(data_stp_ctrl['0'],
                               figname='FigE7_i_left.pdf')
    plot_lhbpfc_paramspace_wta_quantification(data_stp_ctrl['0.4'],
                               figname='FigE7_i_right.pdf')

    return
