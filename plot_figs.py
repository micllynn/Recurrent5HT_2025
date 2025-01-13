from network_model_1a import *

import brian2 as b2


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
