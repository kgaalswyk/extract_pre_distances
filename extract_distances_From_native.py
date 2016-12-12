#!/usr/bin/env python


import pandas as pd
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as pp
import seaborn as sns
import math
import re
import scipy.spatial.distance as sd
#import scipy.stats as st


def load_r2(excel):
    df = pd.read_excel(excel, 'S17C', header=None)
    df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'residue', 'r2', 'j']
    r2 = df.drop(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'j'], 1)
    r2 = r2.dropna()
    r2['residue'] = [int(r) for r in r2['residue']]
    r2 = r2.set_index('residue')
    return r2


def load_data(excel, sheetname):
    df = pd.read_excel(excel, sheetname, header=None)
    df.columns = [
        'residue',
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
    ir = df.drop('g', 1)
    # now drop anything left with NaN
    ir = ir.dropna()
    # now drop the columns we don't need
    ir = ir.drop(['c', 'e', 'f', 'h', 'i', 'j'], 1)
    ir['residue'] = [int(r) for r in ir['residue']]
    ir = ir.set_index('residue')
    return ir


def add_noise(dia, para):
	n = len(dia)
	max_peak = np.max(dia)
	stdev = 0.05 * max_peak
	noise_dia = np.random.normal(0, stdev, n)
	noise_para = np.random.normal(0, stdev, n)
	noise_dia[noise_dia < 0] = 0
	noise_para[noise_para < 0] = 0
	noisy_dia = dia + noise_dia
	noisy_para = para + noise_para
	return noisy_dia, noisy_para

def noise_avg(rs):
    #if rs val > 100, then gamma was zero
    rs = np.array(rs)
    rs[rs>100]= 0
    avg = np.average(rs, axis=0)
    stdev = np.std(rs, axis=0)
    mean = np.mean(rs)
    return avg, mean, stdev


def compute_ratio(ir, name):
    para = ir['paramagnetic'].values
    dia = ir['diamagnetic'].values
    ratios = []
    for i in range(100):
        noisy_dia, noisy_para = add_noise(dia, para)
        ratios.append(noisy_para/noisy_dia)
    ratios = np.array(ratios)
    # normalize so the top 10% of ration have median of 1
    n = len(ratios) // 10
    correction = np.median(np.sort(ratios)[-n:])
    #pp.plot(ratios)
    #pp.axhline(correction)
    #pp.savefig(name+".png")
    #pp.close()
    ratios = ratios / correction
    #ir['ratio'] = ratios
    #ir['ratio'][ir['ratio'] > 1.2] = np.nan
    #ratios[np.isnan(ratios)] = 0 
    return ratios
    #return ir.dropna()


def compute_gamma(ratios, ir):
    #rework to use np array from compute_ratio
    gammas_full = []
    for ratio in ratios:
        gammas = []
        for ra, r2 in zip(ratio, ir['r2']):
            if (ra>1):
                ra=1
            if (ra<0.0):
                ra=1e-5
            def func(gamma):
               return (r2 * math.exp(-gamma * 10e-3) / (r2 + gamma) - ra)
            x = opt.newton(func, 1., maxiter=500)
            gammas.append(x)
        gammas_full.append(gammas)
    #ir['gamma'] = gammas
    #ir[ir['gamma'] < 0] = 0
    #return ir
    return gammas_full


def compute_distance(gammas_full,pre):
    #rework to use np array from compute_ratio
    K = 1.23e-32 * 1e-12 # cm^6 s^-2 * m^6/cm^6 = m^6 s^-2
    tc = 8e-9 # ns
    omega_H = 700e6 # s^-1
    f = 4 * tc + 3 * tc / (1 + (omega_H * tc)**2 )
    rs_full = []
    g = K / (1.0e-9)**6 * f
    for gammas in gammas_full:
        rs = []
        for gamma in gammas:
            if gamma<1e-6:
                gamma = 1e-10
            r6 = K * f / gamma
            r = r6**(1.0 / 6.0) * 1e9
            rs.append(r)
        rs_full.append(rs) 
    #pre['r'] = rs
    return rs_full


def process_bounds(pre, cutoff1, cutoff2):
    def get_lower(r):
        if r < cutoff1:
            return 0.
        elif r > cutoff2:
            return cutoff2
        else:
            return max(0, r - 0.4)

    def get_upper(r):
        if r < cutoff1:
            return cutoff1
        elif r > cutoff2:
            return 999
        else:
            return r + 0.4

    lower = [get_lower(r) for r in pre['r'].values]
    upper = [get_upper(r) for r in pre['r'].values]
    pre['lower'] = lower
    pre['upper'] = upper
    return pre


def pdb_distances(pdb_file, ds_name, resid, r2):
	res_pdb = []
	out_file = open(ds_name+"_pdb_distances.txt", 'w')
	out_file.write('{:7}   {:6}    {:6}  {:4}  {:6}   {:8}    {:8}     {:9}\n'.format(resid+'preID', resid+'OND', resid+'ResID', resid+'N', resid+'PDB',
		resid+'Below', resid+'Above', resid+'Middle'))
	with open(pdb_file) as in_file:
		for line in in_file:
			line = line.split()
			if(line[0]=='ATOM' or line[0]=='HETATM'):
				if(int(line[5]) == int(resid) and line[2]== 'N'):
					res_pdb = ([float(line[6]), float(line[7]), float(line[8])])
					break
	for r, row in r2.iterrows():
		with open(pdb_file) as fp:
			for line in fp:
				line = line.split()
				if(line[0]=='ATOM' or line[0]=='HETATM'):
					if(int(line[5]) == r and line[2]== 'N'):
						r_pos =([float(line[6]), float(line[7]), float(line[8])])
						dist = sd.euclidean(r_pos,res_pdb)
						frmt = '{:3d}        OND       {:3d}       N    {:.3f}      {:.3f}       {:.3f}          {:.3f}\n'
						if(0<row['lower']<2 and 1.2<row['upper']<999):
							out_file.write(frmt.format(int(resid), int(r), dist/10, math.nan , math.nan , row['r']))
						if(row['lower']==0 or row['upper']==1.2):
							out_file.write(frmt.format(int(resid), int(r), dist/10, row['r'], math.nan, math.nan ))
						if(row['lower']==2 or row['upper']==999):
							out_file.write(frmt.format(int(resid), int(r), dist/10, math.nan, row['r'], math.nan))
							
	out_file.close()					


def ond_to_pre_compare(ond_dist, ds_name, resid, r2, stdev):
	res = ond_dist[resid+"_dist"]
	out_file = open(ds_name+"pdb_distances.txt", 'w')
	out_file.write('{:7}   {:6}  {:6}   {:8}    {:8}     {:9}     {:8}     {:8}     {:8}\n'.format(resid+'preID', resid+'ResID', resid+'PDB',
		resid+'Below', resid+'Below+-', resid+'Above', resid+'Above+-', resid+'Middle', resid+'Middle+-'))
	for (r, row), std_dev in zip(r2.iterrows(), stdev):
		dist = ond_dist.loc[ond_dist['resid']== str(r), resid+'_dist'].values
		if(dist):
			frmt = '{:3d}        {:3d}       {:.3f}      {:.3f}       {:.3f}          {:.3f}       {:.3f}       {:.3f}       {:.3f}\n'
			if(0<row['lower']<2 and 1.2<row['upper']<999):
				out_file.write(frmt.format(int(resid), int(r), dist[0], math.nan, int(0), math.nan, int(0), row['r'], float(std_dev)/10))
			if(row['lower']==0 or row['upper']==1.2):
				out_file.write(frmt.format(int(resid), int(r), dist[0], row['r'], float(std_dev)/10, math.nan, int(0), math.nan, int(0)))
			if(row['lower']==2 or row['upper']==999):
				out_file.write(frmt.format(int(resid), int(r), dist[0], math.nan, int(0), row['r'], float(std_dev)/10, math.nan, int(0)))
	out_file.close()		

def ond_distances(pdb_file):
	ond_id = []
	ond_ndist = []
	ond_pos = []
	n_pos = []
	resids = []

	with open(pdb_file) as in_file:
		for line in in_file:
			line=line.split()
			if(len(line)>1):
				if(line[2]=='OND'):
					ond_id.append([line[3], line[4], line[5], line[6], line[7]])
				if(line[2]=='N'):
					if(int(line[4])>146):
						line[4]= int(line[4])+51
					ond_ndist.append([line[3], int(line[4]), line[5], line[6], line[7]])
	for n in range(0, len(ond_id)):
		ond_pos.append([float(ond_id[n][2]), float(ond_id[n][3]), float(ond_id[n][4])])
	for m in range(0,len(ond_ndist)):
		n_pos.append([float(ond_ndist[m][2]), float(ond_ndist[m][3]), float(ond_ndist[m][4])])
		resids.append(ond_ndist[m][1])

	ond_dists = [{"resname":ond_ndist[x][0], "resid":str(int(ond_ndist[x][1])+4), 
		str(int(ond_id[0][1])+4)+"_dist":sd.euclidean(ond_pos[0], n_pos[x])/10, str(int(ond_id[1][1])+4)+"_dist":sd.euclidean(ond_pos[1], n_pos[x])/10, 
		str(int(ond_id[2][1])+4)+"_dist":sd.euclidean(ond_pos[2], n_pos[x])/10, str(int(ond_id[3][1])+4)+"_dist":sd.euclidean(ond_pos[3], n_pos[x])/10, 
		str(int(ond_id[4][1])+4)+"_dist":sd.euclidean(ond_pos[4], n_pos[x])/10, str(int(ond_id[5][1])+4)+"_dist":sd.euclidean(ond_pos[5], n_pos[x])/10, 
		str(int(ond_id[6][1])+4)+"_dist":sd.euclidean(ond_pos[6], n_pos[x])/10, str(int(ond_id[7][1])+4)+"_dist":sd.euclidean(ond_pos[7], n_pos[x])/10,
		str(int(ond_id[8][1])+4)+"_dist":sd.euclidean(ond_pos[8], n_pos[x])/10} for x in range(len(ond_ndist))]

	df = pd.DataFrame(ond_dists, resids)
	return df

excel = pd.ExcelFile('CaCaM2smMLCK_06102015.xls')
r2 = load_r2(excel)

ds_names = ['S17C', 'T34C', 'N42C', 'N53C', 'R86C', 'T110C', 'E127C', 'Q143C']#, 'C149']
ond_dist = ond_distances('with_OND.pdb')
#pdb_distances('cam2smmlck.pdb', DS_NAME, resid, pre)
for DS_NAME in ds_names:
	resid = re.sub(r'[^\d]', '', DS_NAME)
	pre = load_data(excel, DS_NAME)
	pre = pd.merge(pre, r2, left_index=True, right_index=True)
	
	ratios = compute_ratio(pre, DS_NAME)
	gammas = compute_gamma(ratios, pre)
	rs = compute_distance(gammas, pre)
	ratios_full, avg, stdev = noise_avg(rs)
	pre['r'] = ratios_full

	#include func to get avg, stdev, error bars, yada yada from ratios
	pre = process_bounds(pre, cutoff1=1.2, cutoff2=2.0)

	ond_to_pre_compare(ond_dist, DS_NAME, resid, pre, stdev)
