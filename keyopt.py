import numpy as np
import re
import itertools
from pathlib import Path
import multiprocessing as mp

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.cm import ScalarMappable 

from scipy.optimize import quadratic_assignment

dir = Path(__file__).resolve().parent
DIR_DATA = dir / 'data'
DIR_SOL = dir / 'solutions'

AZ = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '_'*6
_AZ = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ012345'

IJ = np.array(list(itertools.product(range(4), range(8))))
IJ2 = np.array(list(itertools.combinations_with_replacement(IJ, 2)))

_P0 = np.arange(32)
P0 = _P0.reshape(4,8)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_permutation(c):
  idx_inv = np.array([ _AZ.index(x) for x in c ])
  idx = np.empty_like(idx_inv)
  idx[idx_inv] = np.arange(32)
  return idx

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def convert_ngrams():
  # https://norvig.com/mayzner.html
  # English Letter Frequency Counts:
  # Mayzner Revisited
  # or
  # ETAOIN SRHLDCU

  ngrams = list()
  data = None

  with open(DIR_DATA / 'ngrams-all.tsv', 'r') as fp:
    for line in fp.readlines():
      row = line.split('\t')

      l = row[0].strip()

      if re.fullmatch('\d+-gram', l):
        if data is not None:
          ngrams.append(data)

        data = []

      else:
        data.append((l, float(row[1])))

  gram1 = np.zeros((32,), dtype = np.float64)

  for g, f in ngrams[0]:
    gram1[AZ.index(g)] = f

  gram2 = np.zeros((32,32), dtype = np.float64)

  for g, f in ngrams[1]:
    a,b = g
    gram2[AZ.index(a), AZ.index(b)] = f

  gram3 = np.zeros((32,32,32), dtype = np.float64)

  for g, f in ngrams[2]:
    a,b,c = g
    gram3[AZ.index(a), AZ.index(b), AZ.index(c)] = f

  np.save(DIR_DATA / 'ngrams-1', gram1)
  np.save(DIR_DATA / 'ngrams-2', gram2)
  np.save(DIR_DATA / 'ngrams-3', gram3)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def dist_metric(x0, y0, x1, y1, a = 1):
  dx = x1 - x0
  dy = y1 - y0
  # elliptic metric, representing spread of fingers reducing lateral movement
  return (dx**2 + (dy/a)**2)**0.5

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def objective(p, W, D):
  return np.trace(W.T @ D[p][:,p])

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _qab(args):
  rank, sols, W, D, niter = args

  for i in range(niter):

    # NOTE: the solution here is a local greedy algorithm, and depends on 
    # the (random) initialization.
    res = quadratic_assignment(
      W, 
      D,
      method = '2opt' )

    if res.fun < sols[-1][0]:
      print(f"{rank}-{i}: {res.fun}")
      sols.append((res.fun, res.col_ind))

    else:
      print(f"{rank}-{i}: --")

  return sols

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def quadratic_assignment_brute(W, D, niter, nproc ):

  res = quadratic_assignment(
    W, 
    D,
    method = 'faq',
    options = dict(
      maxiter = int(1e6),
      tol = 1e-9 ))

  print(res)

  sols = [(res.fun, res.col_ind)]

  _niter = (niter+nproc-1)//nproc

  with mp.Pool(nproc) as pool:
    for _sols in pool.map(_qab, [(i, sols, W, D, _niter) for i in range(nproc)] ):
      for sol in _sols:
        if sol[0] < sols[-1][0]:
          sols.append(sol)

  return sols

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_sol(_p, W, D, out_dir = Path()):
  obj = objective(_p, W, D)

  _pinv = np.empty_like(_p)
  _pinv[_p] = _P0

  p = _p.reshape(4, 8)
  pinv = _pinv.reshape(4, 8)

  _W = np.zeros(IJ2.shape[:1])

  for i, (a, b) in enumerate(IJ2):
    if np.all(a == b):
      continue

    ka = pinv[a[0], a[1]]
    kb = pinv[b[0], b[1]]

    _W[i] = W[ka, kb] + W[kb, ka]
    # print(f"{AZ[ka]} - {AZ[kb]}: {_W[i]}")

  perm = ''.join([AZ[i] for i in _pinv])

  scale_lin = np.linspace(0, 1, 256)
  scale_cos = 1 - 0.5*( 1 + np.cos(np.linspace(0, np.pi, 256)) )

  norm_d = plt.Normalize(0, 1.75)
  cmap_d = np.zeros((256,4))
  cmap_d[:,1] = scale_lin[::-1]
  cmap_d[:,2] = scale_lin[::-1]
  cmap_d[:,3] = 1
  cmap_d = ListedColormap(cmap_d)

  norm_n1 = plt.Normalize(0, g1.max())
  # cmap_n1 = plt.get_cmap('viridis')
  cmap_n1 = np.zeros((256,4))
  cmap_n1[:,0] = scale_lin
  cmap_n1[:,1] = scale_lin
  cmap_n1[:,3] = 1
  cmap_n1 = ListedColormap(cmap_n1)

  norm_n2 = plt.Normalize(0, _W.max())

  cmap_n2 = np.zeros((256,4))
  cmap_n2[:,0] = 1
  cmap_n2[:,3] = 0.8*scale_cos
  cmap_n2 = ListedColormap(cmap_n2)

  fig, ax = plt.subplots( figsize = (10, 4) )
  fig.suptitle(f"objective: {obj:.4f}\n{perm}")

  # im = ax.imshow(g1[pinv], cmap = cmap_n1, norm = norm_n1)
  im = ax.imshow(r, interpolation = 'bicubic', cmap = cmap_d, norm = norm_d)

  cb = fig.colorbar(
    ScalarMappable(cmap = cmap_n1, norm = norm_n1),
    fraction = 0.1*0.5, 
    ax = ax, 
    ticks = np.linspace(0, norm_n1.vmax, 5) )

  cb.ax.set_yticklabels([f'{x:.1%}' for x in cb.ax.get_yticks()])

  lc = LineCollection(
    IJ2[:,:,::-1], 
    cmap = cmap_n2,
    norm = norm_n2,
    linewidth = 2,
    array = _W )

  # lc.set_array(_W)
  # lc.set_linewidth(2)
  line = ax.add_collection(lc)

  cb = fig.colorbar(
    lc, 
    fraction = 0.1*0.5, 
    ax = ax,
    ticks = np.linspace(0, norm_n2.vmax, 5) )

  cb.ax.set_yticklabels([f'{x:.1%}' for x in cb.ax.get_yticks()])

  for i in range(4):
    for j in range(8):
      k0 = pinv[i,j]
      ax.text(
        j, i, 
        AZ[k0], 
        ha = "center", 
        va = "center", 
        color = "white",
        bbox = dict(
          facecolor = [0,0,0,1],
          edgecolor = cmap_n1(norm_n1(g1[k0])),
          linewidth = 4,
          pad = 0.5,
          boxstyle = 'circle' ) )

  ax.set_xticks(np.arange(8))
  ax.set_yticks(np.arange(4))

  fname = out_dir / f'keyopt-{int(1000*obj)}-{perm}.svg'
  print(fname)
  plt.savefig(fname)
  # plt.show()
  plt.cla()
  plt.clf()
  plt.close('all')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':

  # number of restart iterations in global optimization
  niter = 1000
  nproc = 4

  # elliptic axis in horizontal direction 
  # larger value reduces cost of horizontal translation
  # for 1-gram
  a1 = 2
  # for 2-gram
  a2 = 1

  # weight (frequency) of 2-grams
  g2_fac = 1.0

  # only needed to extract values from tsv, saved as the npy files
  # convert_ngrams()

  g1 = np.load(DIR_DATA / 'ngrams-1.npy')
  # Normalize to give sum_A P(A) -> 1
  g1 = g1 / g1.sum()

  g2 = np.load(DIR_DATA / 'ngrams-2.npy')
  # Normalize columns to give sum_A P(B|A) -> P(B)
  cg2 = g2.sum(axis = 0)
  cg2 = np.where(cg2 == 0.0, 1, cg2)
  g2 = g2 * ( g1 / cg2 )[None,:]

  g3 = np.load(DIR_DATA / 'ngrams-3.npy')
  cg3 = g3.sum(axis = (0,1))
  cg3 = np.where(cg3 == 0.0, 1, cg3)
  g3 = g3 * ( g1 / cg3 )[None,None,:]

  # print(g1)
  # print(g2)
  print(g1.shape)
  print(g2.shape)

  x, y = np.meshgrid( np.arange(4), np.arange(8), indexing = 'ij')
  _x = x.ravel()
  _y = y.ravel()

  # 1-gram distance from 'centered' position
  # NOTE: offset to remove reflection symmetry (degenerate minimums)
  r = dist_metric(1.55, 3.45, x, y, a1)

  d = r.copy()
  d[3,:3] = 1e6
  d[3,-3:] = 1e6
  
  D = np.diag(d.ravel())

  # 2-gram distances
  for i, j in itertools.product(_P0, repeat = 2):
    D[i,j] += dist_metric(_x[i], _y[i], _x[j], _y[j], a2)

  W = np.diag(g1)

  W[:] += g2_fac * g2

  # these are just the 'control' solutions for comparison
  nominal = get_permutation("ABCDEFGHIJKLMNOPQRSTUVWX012YZ345")
  qwerty = get_permutation("QWERTYUIASDFGHJOZXCVBNMP012KL345")

  _ps = [
    (objective(nominal, W, D), nominal), 
    (objective(qwerty, W, D), qwerty)]

  _ps.extend(quadratic_assignment_brute(W, D, niter = niter, nproc = nproc))

  for obj, _p in _ps:
    plot_sol(_p, W, D, out_dir = DIR_SOL)

