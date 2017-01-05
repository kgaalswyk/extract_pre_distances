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
    df = pd.read_excel(excel, 'S17C', header=None, converters={'H': int})
    df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'resid', 'r2', 'j']
    r2 = df.drop(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'j'], axis=1)
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


def compute_crystal_distances(pdb_file):
    onds = []
    nitrogens = []
    with open(pdb_file) as in_file:
        for line in in_file:
            line = line.split()
            if(len(line) > 1):
                # if it is an OND store the resname, resid, x, y, z
                # resid is +4 to match NMR data
                if(line[2] == 'OND'):
                    onds.append([line[3], int(line[4]) + 4, float(line[5]),
                                 float(line[6]), float(line[7])])

                # if it is an amide nitrogen, store the same things
                if(line[2] == 'N'):
                    # the pdb file has sequential indexing, but the nmr data
                    # has an offset between the protein and the peptide
                    if(int(line[4]) > 146):
                        line[4] = int(line[4]) + 51

                    nitrogens.append([line[3], int(line[4]) + 4, float(line[5]),
                                      float(line[6]), float(line[7])])

    # get the resids and restypes and create dataframe
    resids = [x[1] for x in nitrogens]
    restypes = [x[0] for x in nitrogens]
    df = pd.DataFrame({'resid': resids, 'restype': restypes})

    # compute the distances between the ONDs and nitrogens
    ond_pos = np.array([[x[2], x[3], x[4]] for x in onds])
    n_pos = np.array([[x[2], x[3], x[4]] for x in nitrogens])
    dists = sd.cdist(ond_pos, n_pos, 'euclidean') / 10.  # convert from Angstrom to nm

    # put the distances into columns of our dataframe
    for index, ond in enumerate(onds):
        column_name = 'r_crystal_{}'.format(ond[1])
        df[column_name] = dists[index, :]

    return df


def main():
    # Load the excel file
    excel = pd.ExcelFile('CaCaM2smMLCK_06102015.xls')

    # Compute the distances to paramagnetic centers from the pdb file
    # and return them in a dataframe
    df = compute_crystal_distances('with_OND.pdb')

    # Add r2 to the dataframe
    r2 = load_r2(excel)
    df = df.merge(r2, how='outer', on='resid')

    # Loop over the data sets and compare the distances predicted from the NMR data
    # to the distances from the PDB file.
    ds_names = ['S17C', 'T34C', 'N42C', 'N53C', 'R86C', 'T110C', 'E127C', 'Q143C', 'C149']
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
        df = df.merge(pre, how='outer', on='resid')

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
                tools=TOOLS)#,
                # x_range=(0, 4.5),
                # y_range=(0, 4.5))

        # Draw "good" and "bad" boxes
        p.patch([0, 2.0, 2.0, 1.2, 0], [0, 0, 2.4, 1.6, 1.6], color='green', alpha=0.1)
        p.patch([0, 1.2, 2.0, 2.0, 1.2, 0], [1.6, 1.6, 2.4, 4.5, 4.5, 4.5], color='red', alpha=0.1)

        # Draw +/- 0.4 angstrom lines.
        p.line([0, 4.5], [0.4, 4.9], color='grey')
        p.line([0, 4.5], [-0.4, 4.1], color='grey')

        # Plot the predicted vs actual distance.
        # The plots will be linked because they all share the same
        # datasource.
        p.circle('r_nmr_mean_{}'.format(resid),
                 'r_crystal_{}'.format(resid),
                 source=source,
                 name='distance')
        # p.circle('para_{}'.format(resid),
        #          'dia_{}'.format(resid),
        #          source=source,
        #          name='distance')

        # Set the tool-tips
        hover = p.select(dict(type=HoverTool))
        hover.tooltips = [
            ('resid', '@resid'),
            ('restype', '@restype'),
            ('pre', '@r_nmr_mean_{}'.format(resid)),
            ('xtal', '@r_crystal_{}'.format(resid)),
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
