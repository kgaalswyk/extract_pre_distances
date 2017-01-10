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


def load_data(excel, sheetname):
    df = pd.read_excel(excel, sheetname, header=None, converters={'A': int})
    df.rename(columns={
                       0: 'resid',
                       27: 'gamma',
                      },
              inplace=True
    )
    # only keep the columns we care about
    df = df[['resid', 'gamma']]
    # now drop anything left with NaN
    df = df.dropna()
    return df


def compute_distances(gamma):
    return [compute_single_distance(g) for g in gamma]


def compute_single_distance(gamma):
    '''
    Compute the distance from gamma

    Formulas and approach from:
    W.D. Van Horn, A.J. Beel, C. Kang, C.R. Sanders, The impact of window
    functions on NMR-based paramagnetic relaxation enhancement
    measurements in membrane proteins, BBA, 2010, 1798: 140-149
    '''
    K = 1.23e-32 * 1e-12 # cm^6 s^-2 * m^6/cm^6 = m^6 s^-2
    tc = 8e-9 # ns
    omega_H = 600e6 # s^-1
    f = 4 * tc + 3 * tc / (1 + (omega_H * tc)**2 )

    if gamma <= 0:
        distance = np.NAN
    else:
        distance = (K * f / gamma)**(1.0 / 6.0) * 1e9
    return distance


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
    excel = pd.ExcelFile('CaCaM2smMLCK_12222016.xls')

    # Compute the distances to paramagnetic centers from the pdb file
    # and return them in a dataframe
    df = load_crystal_distances()

    # Loop over the data sets and compare the distances predicted from the NMR data
    # to the distances from the PDB file.
    # For now, only T110C has the next two-point data
    ds_names = ['T110C']
    ond_resids = []
    for ds_name in ds_names:
        # figure out the name of our new columns
        ond_resid = re.sub(r'[^\d]', '', ds_name)
        ond_resids.append(ond_resid)
        gamma_name = 'gamma_{}'.format(ond_resid)
        r_name = 'r_nmr_{}'.format(ond_resid)

        # Load in the PRE dataset and merge into the data frame
        pre = load_data(excel, ds_name)
        pre = pre.rename(columns={'gamma': gamma_name})
        pre = pre.set_index('resid')
        df = df.merge(pre, how='outer', left_index=True, right_index=True)

        # compute the distances based on gamma
        gamma = df[gamma_name]
        distances = compute_distances(gamma)
        df[r_name] = distances


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
    for resid, mut_name in zip(ond_resids, ds_names):
        p = figure(plot_width=250, plot_height=250,
                tools=TOOLS)#,
                # x_range=(0, 4.5),
                # y_range=(0, 4.5))

        # Draw "good" and "bad" boxes
        # p.patch([0, 2.0, 2.0, 1.2, 0], [0, 0, 2.4, 1.6, 1.6], color='green', alpha=0.1)
        # p.patch([0, 1.2, 2.0, 2.0, 1.2, 0], [1.6, 1.6, 2.4, 4.5, 4.5, 4.5], color='red', alpha=0.1)
        p.patch([0, 1.5, 1.5, 0], [0, 0, 1.9, 1.9], color='green', alpha=0.1)
        p.patch([0, 1.5, 1.5, 0], [1.9, 1.9, 5.0, 5.0], color='red', alpha=0.1)

        # Draw +/- 0.4 angstrom lines.
        p.line([0, 4.5], [0.4, 4.9], color='grey')
        p.line([0, 4.5], [-0.4, 4.1], color='grey')

        # Plot the predicted vs actual distance.
        # The plots will be linked because they all share the same
        # datasource.
        p.circle('r_nmr_{}'.format(resid),
                 'avg_dist_{}'.format(resid),
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
            ('pre', '@r_nmr_{}'.format(resid)),
            ('xtal', '@avg_dist_{}'.format(resid)),
            ('gamma', '@gamma_{}'.format(resid))
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
