import numpy as np
import re
import itertools
import functools
from pathlib import Path
import multiprocessing as mp

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.cm import ScalarMappable 

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
def qap_objective(C, F, D, p):
  r"""Objective function for the quadratic assignment problem

  .. math::

    trace(C p^T + F^T p D p^T)

  Parameters
  ----------
  C : array[n,n]
    Cost to place facility i at location j
  F : array[n,n]
    Flow between facilities i and j
  D : array[n,n]
    Distance of transition between locations i and j
  p : array[n]
    The placement permutation vector
  """

  return np.einsum('ij,ij', F, D[p][:,p]) + np.einsum('ii', C[:,p])

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def assignment_2opt(n, f, p = None):
  """https://github.com/scipy/scipy/blob/v1.9.3/scipy/optimize/_qap.py
  """

  rng = np.random.default_rng()
  p0 = np.arange(n)

  if p is None:
    p = rng.permutation(p0)
  
  c = f(p)
  cref = c

  print(f"  cref: {cref:.2e}")

  cbest = c
  pbest = p

  if cref == 0:
    return c, p

  n_iter = 0

  k = 8.0
  kmod = 10
  kfac = 1.25
  kmax = 2**32

  cost = np.zeros((n,n), dtype = np.float64)
  
  while True:
    n_iter += 1
    cost[:] = -1.0

    if n_iter % kmod == 0 and k < kmax:
      k *= kfac

    for i, j in itertools.combinations_with_replacement(p0, 2):
      if i == j:
        continue

      p[i], p[j] = p[j], p[i]
      _c = f(p)

      if _c < cbest:
        cbest = _c
        pbest = p.copy()

      cost[i,j] = _c
      p[i], p[j] = p[j], p[i]

    m = cost < 0.0
    x = k*(cost - c)/cref
    m |= x < -50

    x = np.clip(x, -50, 50)
    prob = np.exp(-x)
    prob = np.where(m, 0.0, prob)
    s0 = prob.max()

    if s0 == 0.0:
      break

    prob /= s0
    prob /= prob.sum()

    if k < kmax:
      i = rng.choice(n*n, p = prob.ravel())
    else:
      i = np.argmax(prob.ravel())

    i, j = divmod(i, n)

    p[i], p[j] = p[j], p[i]
    _c = f(p)

    if k >= kmax and _c >= c:
      p[i], p[j] = p[j], p[i]
      break

    if _c < 2*c:
      c = _c
    else:
      p[i], p[j] = p[j], p[i]

    if n_iter % kmod == 0:
      print(f"  {n_iter}: {int(np.log2(k))}, ({i},{j}) -> {c/cref:.2e}")
      # plt.imshow(prob)
      # plt.title(f"{n_iter}: {int(np.log2(k))}, ({i},{j}) -> {c/cref:.2e}")
      # plt.show()

  if cbest < c:
    c = cbest
    p = pbest

  print(f"  best: {c:.2e}")

  return c, p

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def assignment_2opt_shuffle(n, m, niter, f):
  rng = np.random.default_rng()

  p0 = np.arange(n)
  c, p = assignment_2opt(n, f)
  cref = c

  print(f"cref: {cref:.2e}")
  
  for i in range(niter):
    idx = rng.choice(n, m, replace = False)
    _p = p.copy()
    _p[idx] = rng.permutation(p[idx])

    _c, _p = assignment_2opt(n, f, _p)

    if _c < c:
      c = _c
      p = _p

    print(f"{i+1}: {c/cref:.2e}")

  print(f"best: {c:.2e}")

  return c, p

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_sol(_p, C, F, D, ref, out_dir = Path()):
  obj = qap_objective(C, F, D, _p) / ref

  _pinv = np.empty_like(_p)
  _pinv[_p] = _P0

  p = _p.reshape(4, 8)
  pinv = _pinv.reshape(4, 8)

  _F = np.zeros(IJ2.shape[:1])

  for i, (a, b) in enumerate(IJ2):
    if np.all(a == b):
      continue

    ka = pinv[a[0], a[1]]
    kb = pinv[b[0], b[1]]

    _F[i] = F[ka, kb] + F[kb, ka]
    # print(f"{AZ[ka]} - {AZ[kb]}: {_F[i]}")

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

  norm_n2 = plt.Normalize(0, _F.max())

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
    array = _F )

  # lc.set_array(_F)
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

  fname = out_dir / f'keyopt-{int(1000*obj):04}-{perm}.svg'
  print(fname)
  plt.savefig(fname)
  # plt.show()
  plt.cla()
  plt.clf()
  plt.close('all')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':

  # number of restart iterations in global optimization
  niter = 10
  nproc = 1

  # elliptic axis in horizontal direction 
  # larger value reduces cost of horizontal translation
  # for 1-gram
  a1 = 2
  # for 2-gram
  a2 = 1

  # weight (frequency) of 2-grams
  g2_fac = 1.0

  # weight factor for logical relation between characters
  # NOTE: should be small to tie-break based on lexical ordering
  l2_fac = 0.001

  # Amount of asymmetry to introduce to tie-break mirror symmetries 
  asym_fac = 0.02

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
  r = dist_metric(1.5, 3.5, x, y, a1)

  # NOTE: remove reflection symmetry that causes degenerate minimums
  r -= asym_fac*(x - 1.5)
  r += asym_fac*(y - 3.5)

  d = r.copy()
  # NOTE: forces solution away from bottom left/right 3 keys, reserved for 
  # 2U space, punctuation, and other future use
  d[3,:3] = 1e6
  d[3,-3:] = 1e6
  
  D = np.diag(d.ravel())

  # 2-gram distances
  for i, j in itertools.product(_P0, repeat = 2):
    D[i,j] += dist_metric(_x[i], _y[i], _x[j], _y[j], a2)

  F = np.diag(g1)

  F[:] += g2_fac * g2

  # NOTE: this weight is based on the logical distance between the characters
  # in lexical ording, falling off quickly for well separated characters 
  # e.g. (A,B) has a higher weight than (A,G), but (A,A) -> 0.
  l = (_P0[None, :] - _P0[:, None])**2
  m = l > 0.0
  F[:] += l2_fac * m / np.where(m, l, 1)

  # placement 'cost'
  C = np.zeros((32,32))

  # these are just the 'control' solutions for comparison
  nominal = get_permutation("ABCDEFGHIJKLMNOPQRSTUVWX012YZ345")
  qwerty = get_permutation("QWERTYUIASDFGHJOZXCVBNMP012KL345")

  _ps = [
    (qap_objective(C, F, D, nominal), nominal), 
    (qap_objective(C, F, D, qwerty), qwerty)]

  f = functools.partial(qap_objective, C, F, D)
  _ps.append(assignment_2opt_shuffle(len(F), len(F)//2, niter, f))

  for obj, _p in _ps:
    plot_sol(_p, C, F, D, ref = _ps[0][0], out_dir = DIR_SOL)

