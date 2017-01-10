#!/usr/bin/env python


import pandas as pd
import numpy as np
import scipy.optimize as opt
import math
import re
import scipy.spatial.distance as sd
import random
from bokeh.plotting import figure, output_file, show
from bokeh.models.sources import ColumnDataSource
from bokeh.models import  HoverTool, Label, Span
from bokeh.layouts import gridplot


def load_r2(excel):
    df = pd.read_excel(excel, 'C149', header=None, converters={7: int})
    r2 = df.drop([0, 1, 2, 3, 4, 5, 6, 9], axis=1)
    r2.columns = ['resid', 'r2']
    r2 = r2.dropna()
    return r2


def load_data(excel, sheetname):
    df = pd.read_excel(excel, sheetname, header=None, converters={'A': int})
    df.columns = [
        'resid',
        'diamagnetic',
        'c',
        'paramagnetic',
        'e',
        'f',
        'g',
        'h',
        'i',
        'j']
    # drop 'g' first, because it's all NaN
    df = df.drop('g', axis=1)
    # now drop anything left with NaN
    df = df.dropna()
    # now drop the columns we don't need
    df = df.drop(['c', 'e', 'f', 'h', 'i', 'j'], axis=1)
    return df


def compute_distances(para, dia, r2, n_trials, noise_pct):
    # compute the noise level as a percentage of the highest intensity
    noise_level = noise_pct * np.max(dia)

    means = []
    stds = []
    for p, d, r in zip(para, dia, r2):
        m, s = compute_single_distance(p, d, r, n_trials, noise_level)
        means.append(m)
        stds.append(s)

    return means, stds


def compute_single_distance(para, dia, r2, n_trials, noise_level):
    '''
    Compute the distance from peak intensities.

    Also computes the standard deviation over n_trials, assuming the
    peaks are corrupted by noise with standard deviation of noise_level.

    Formulas and approach from:
    W.D. Van Horn, A.J. Beel, C. Kang, C.R. Sanders, The impact of window
    functions on NMR-based paramagnetic relaxation enhancement
    measurements in membrane proteins, BBA, 2010, 1798: 140-149
    '''
    K = 1.23e-32 * 1e-12 # cm^6 s^-2 * m^6/cm^6 = m^6 s^-2
    tc = 8e-9 # ns
    omega_H = 700e6 # s^-1
    f = 4 * tc + 3 * tc / (1 + (omega_H * tc)**2 )

    distances = []
    for _ in range(n_trials):
        dia_sample = dia + random.gauss(0.0, noise_level)
        para_sample = para + random.gauss(0.0, noise_level)

        ratio = para_sample / dia_sample

        # Clamp ratio to lie between 1e-6 and 0.99
        ratio = 1e-6 if ratio < 1e-6 else ratio
        ratio = 0.99 if ratio > 0.99 else ratio

        func = lambda gamma, r=ratio: r2 * math.exp(-gamma * 10e-3) / (r2 + gamma) - r
        try:
            gamma = opt.newton(func=func, x0=0.1, maxiter=500)
            r = (K * f / gamma)**(1.0 / 6.0) * 1e9
            r = 5 if r > 5 else r
        except (RuntimeError, OverflowError):
            r = np.NaN
        distances.append(r)
    return np.mean(distances), np.std(distances)


def load_crystal_distances():
    AVG_FILE = '../AverageDistances/average_distances.dat'
    df = pd.read_csv(AVG_FILE, sep='\t')
    df.columns = ['spin_label_index', 'resid', 'restype', 'distance']
    groups = df.groupby('spin_label_index')
    output = pd.DataFrame()
    for spin_label, residues in groups:
        column_name = 'avg_dist_{}'.format(spin_label)
        residues = residues.set_index('resid')
        output[column_name] = residues['distance']
        output['restype'] = residues['restype']
    return output


def main():
    # Load the excel file
    excel = pd.ExcelFile('CaCaM2smMLCK_06102015.xls')

    # Compute the distances to paramagnetic centers from the pdb file
    # and return them in a dataframe
    df = load_crystal_distances()

    # Add r2 to the dataframe
    r2 = load_r2(excel)
    r2 = r2.set_index('resid')
    df = df.merge(r2, how='outer', left_index=True, right_index=True)

    # Loop over the data sets and compare the distances predicted from the NMR data
    # to the distances from the PDB file.
    ds_names = ['S17C', 'T34C', 'N42C', 'N53C', 'R86C', 'T110C', 'E127C', 'Q143C']
    ond_resids = []
    for ds_name in ds_names:
        # figure out the name of our new columns
        ond_resid = re.sub(r'[^\d]', '', ds_name)
        ond_resids.append(ond_resid)
        para_name = 'para_{}'.format(ond_resid)
        dia_name = 'dia_{}'.format(ond_resid)
        r_mean_name = 'r_nmr_mean_{}'.format(ond_resid)
        r_std_name = 'r_nmr_std_{}'.format(ond_resid)

        # Load in the PRE dataset and merge into the data frame
        pre = load_data(excel, ds_name)
        pre = pre.rename(columns={'paramagnetic': para_name, 'diamagnetic': dia_name})
        pre = pre.set_index('resid')
        df = df.merge(pre, how='outer', left_index=True, right_index=True)

        # compute the average and standard deviations of the distances
        # based on gaussain noise in the peak intensities
        para = df[para_name]
        dia = df[dia_name]
        r2 = df['r2']

        # We could compute noise, but right now we're not.
        N_TRIALS = 1
        NOISE_PCT = 0.
        means, stds = compute_distances(para, dia, r2,
                                        n_trials=N_TRIALS,
                                        noise_pct=NOISE_PCT)
        df[r_mean_name] = means
        df[r_std_name] = stds

    # Now we will print out a list of VMD selection commands, so that
    # we can visualize what information we have
    for ond_resid in ond_resids:
        distances = df['r_nmr_mean_{}'.format(ond_resid)]
        distances = distances[distances < 1.5]
        close_residues = []
        for resid, dist in distances.iteritems():
            close_residues.append(str(resid - 4))
        print '(resid {ond_index} and name OND) or (resid {close_residues} and name N)'.format(
            ond_index=int(ond_resid) - 4,
            close_residues=' '.join(close_residues))

    # We now have all of the data loaded in one big dataframe,
    # and we're going to use bokeh to plot it. We'll store the
    # output in plot.html
    output_file('plot.html')

    TOOLS = "tap,help,hover"

    # we'll loop over all of the PRE labels, except C149 because
    # it is not present in the PDB file
    df = df.dropna()
    source = ColumnDataSource(data=df)
    plots = []
    for resid, mut_name in zip(ond_resids[:-1], ds_names):
        p = figure(plot_width=250, plot_height=250,
                tools=TOOLS)

        # Draw "good" and "bad" boxes
        p.patch([0, 1.5, 1.5, 0], [0, 0, 1.9, 1.9], color='green', alpha=0.1)
        p.patch([0, 1.5, 1.5, 0], [1.9, 1.9, 5.0, 5.0], color='red', alpha=0.1)


        # Draw +/- 0.4 angstrom lines.
        p.line([0, 4.5], [0.4, 4.9], color='grey')
        p.line([0, 4.5], [-0.4, 4.1], color='grey')

        # Plot the predicted vs actual distance.
        # The plots will be linked because they all share the same
        # datasource.
        p.circle('r_nmr_mean_{}'.format(resid),
                 'avg_dist_{}'.format(resid),
                 source=source,
                 name='distance')

        # Set the tool-tips
        hover = p.select(dict(type=HoverTool))
        hover.tooltips = [
            ('resid', '@resid'),
            ('restype', '@restype'),
            ('pre', '@r_nmr_mean_{}'.format(resid)),
            ('xtal', '@avg_dist_{}'.format(resid)),
            ('I_para', '@para_{}'.format(resid)),
            ('I_dia', '@dia_{}'.format(resid)),
            ('r2', '@r2')
        ]
        hover.names = ['distance']

        # Add a label
        label = Label(x=0.6, y=4.0, text=mut_name, text_color='grey', text_align='center')
        p.add_layout(label)

        plots.append(p)

    grid = gridplot(plots, ncols=3)
    show(grid)


if __name__ == '__main__':
    main()
