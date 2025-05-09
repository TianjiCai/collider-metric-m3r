import numpy as np
import ot  # PyOT library: https://pythonot.github.io/index.html
import itertools

from tqdm import tqdm
import concurrent.futures as cf

def get_clean_phi(phi, range='unsigned'):
    '''
    in place of  [- (2*np.pi - delta_phi) if delta_phi > np.pi
                 else 2*np.pi + delta_phi if delta_phi < - np.pi
                 else delta_phi]
    ------
    phi, result: np.array
    '''
    assert range in ['unsigned', 'signed']
    
    result = np.where(phi > np.pi,- (2 * np.pi - phi),np.where(phi < -np.pi, 2*np.pi+phi, phi))
    
    if range == 'unsigned':
      result += np.pi
      
    return result

def get_W2_Kin(jet1, jet2, numItermax=100000, return_gamma=False):
    '''
    Compute the exact W2 distance between two jets, 
    with the ground space being the kinematic space (y, phi), and pt (normalized) or charge as mass.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    jet1, jet2: np.array of shape (nparticles, 4), where the 4-dim is (pt, y, phi, chg).
    mass: str; if 'pt', then mass of each particle is its pt; 
               if 'chg', then mass is the charge of the particle, scaled up by the following upscale_factor.
    upscale_factor: float or None; only if mass='chg', then increase all charge values by a certain amount
                    to avoid zero and negative mass (recommended: 2; does not matter much); default None. 
    numItermax: int; default 100000; hyperparameter in computing W2 distance.
    return_gamma: boolean; default False, i.e., do not return the transport matrix.
    
    OUTPUTS:
    Gamma: None or np.array of shape (nparticles, nparticles)
    OT: float
    '''

    #separate pts, kinematic coordinates, charges
    jet1_pts, jet1_coords = jet1[:,0], jet1[:,1:3]
    jet2_pts, jet2_coords = jet2[:,0], jet2[:,1:3]
    
    # remove zero-padding 
    jet1_pts, jet1_coords = jet1_pts[jet1_pts>0], jet1_coords[jet1_pts>0]
    jet2_pts, jet2_coords = jet2_pts[jet2_pts>0], jet2_coords[jet2_pts>0]
    
    # ground distance
    d_phis = np.pi - np.abs(np.pi - np.abs(jet1_coords[:,1,None]-jet2_coords[:,1]))
    squareDist = (jet1_coords[:,0,None]-jet2_coords[:,0])**2 + d_phis**2
    
    # normalize total pt to be one
    jet1_pts, jet2_pts = jet1_pts/np.sum(jet1_pts), jet2_pts/np.sum(jet2_pts)
    
    # ot computation
    OT = (ot.emd2(jet1_pts, jet2_pts, squareDist, numItermax=numItermax)) ** (1/2)
    Gamma = None
    if return_gamma:
        Gamma = ot.emd(jet1_pts, jet2_pts, squareDist, numItermax=numItermax)
    return Gamma, OT

# def calcOTDistance(xEvents, yEvents, OTSCHEME, COSTSCHEME, kwargs={}, Matrix = False):
#   """
#   Solve the optimal transport problem and find the 2-Wasserstein distance
#   for a set of source events (xEvents) and target events (yEvents) for a
#   given ground cost function.

#   Inputs:
#         xEvents:  Array of sample of x-type events; shape (N, 19, 3)
#         yEvents:  Array of sample of y-type events; shape (N, 19, 3)
#        OTSCHEME:  Determines what scheme will be used to calculate the OT distance.
#                   Note the exact meaning varies somewhat depending on the choice of COSTSCHEME.
#                   It's a dictionary of 3 booleans cooresponding to whether the PT is normalized,
#                   whether the OT calculation is balanced, and whether the zero padding should be
#                   removed. Namely,
#                   OTSCHEME['normPT']      ==True:  Means that the pT should be normalized;
#                                          ==False:  Means that the pT should be unnormalized
#                   OTSCHEME['balanced']    ==True:  Means that the OT calculation should be balanced;
#                                          ==False:  Means that the OT calculation should be unbalanced
#                   OTSCHEME['noZeroPad']   ==True:  Means that the zero padding should be removed;
#                                          ==False:  Means that the zero padding should be kept
#                   OTSCHEME['individualOT']==True:  Means that the OT calculation is done on each species separately;
#                                          ==False:  Means that the OT calculation is done ignoring species type
#      COSTSCHEME:  Determines what scheme will be used to calculate the ground cost matrix. Options are:
#                   - 1DpT: Ground space is pT only
#                   - 2D:   Ground space is 2D (eta,phi); note mass is pT
#                   - 3D:   Ground space is 3D (pT,eta,phi); note mass is uniform
#          kwargs:  Dictionary containing keyword arguments for unbalanced OT calculation; default is an empty dictionary
#          MATRIX:  Whether to return wXY as an (N,N) array or as a (N*N,) array

#   Outputs
#      wXY:  Array of the corresponding (squared) 2-Wasserstein distances
#   """

#   #-- Sanity checks on inputs --#

#   # Check schemes
#   assert checkSCHEMES(OTSCHEME, COSTSCHEME)==True
#   if OTSCHEME['balanced']==False:
#     assert all([x in list(kwargs.keys()) for x in ['div', 'reg_m'] ])

#   # Get number of signal and background events #! Make more general later to handle different number of x and y events
#   assert(xEvents.shape[0] == yEvents.shape[0])
#   N = xEvents.shape[0]

#   #-- Preliminaries --#
#   # Create objects for outputs
#   wXY = np.zeros(shape=(N,N))

#   #-- Loop over pairs of events --#
#   # We use itertools to make looping more efficient (i.e. do double for loops)
#   dummyArr = np.arange(N)
#   wXY_list = []
#   for (i, j) in tqdm(itertools.product(dummyArr, dummyArr), total=len(dummyArr)**2): #! Implement diagonal+upper-triangle restriction to save on computation time
#     #-- Get source and target data points--#
#     # Remove zero-padding if specified
#     if OTSCHEME['noZeroPad']==True:
#       sMask = ~np.all(xEvents[i, :, :] == 0., axis=1) # Mask to remove zero rows
#       tMask = ~np.all(yEvents[j, :, :] == 0., axis=1)

#       xs = xEvents[i, :, :][sMask] # "source data points"
#       xt = yEvents[j, :, :][tMask] # "target data points"
#     else:
#       xs = xEvents[i, :, :] # "source data points"
#       xt = yEvents[j, :, :] # "target data points"

#     #-- Get masses --#
#     a,b = getMasses(xs, xt, OTSCHEME, COSTSCHEME)

#     #-- Normalize pT coordinate if specified --#
#     #! Might be more efficient to do this before getMasses() but then will need to change getMasses()
#     if OTSCHEME['normPT']==True:
#       totPTs, totPTt = xs[:,0].sum(), xt[:,0].sum()
#       xs[:,0], xt[:,0] = xs[:,0]/totPTs, xt[:,0]/totPTt

#     #-- Get cost function and append to list --#
#     cxy = calcGroundCostMatrix(xs, xt, COSTSCHEME)
#     # print(cxy.shape)
#     # print(a.shape)
#     # print(b.shape)

#     #-- Calculate the unbalanced Wasserstein distance --#
#     if OTSCHEME['balanced']==False:
#       if j >= i:
#         wXY[i,j] = ot.unbalanced.mm_unbalanced2(a, b, cxy, reg_m=kwargs['reg_m'], div=kwargs['div'])
#         wXY_list.append(wXY[i,j])
#       else:
#         wXY[i,j] = wXY[j,i]
#     else:
#       if j >= i:
#         wXY[i,j] = ot.emd2(a, b, cxy)
#         wXY_list.append(wXY[i,j])
#       else:
#         wXY[i,j] = wXY[j,i]

#   if Matrix == True:
#     return wXY

#   else:
#     return np.asarray(wXY_list)

def calcOTDistance(events):
  N = events.shape[0]
  wXY = np.zeros(shape=(N,N))
  for i, j in tqdm(itertools.combinations_with_replacement(range(N), 2), total=N*(N+1)//2):
    _, wXY[i,j] = get_W2_Kin(events[i], events[j])
    wXY[j,i] = wXY[i,j]
  return wXY

# def calcOTDistance_non_square(xEvents, yEvents, OTSCHEME, COSTSCHEME, kwargs={}, Matrix = False):
#   """
#   Solve the optimal transport problem and find the 2-Wasserstein distance
#   for a set of source events (xEvents) and target events (yEvents) for a
#   given ground cost function.

#   Inputs:
#         xEvents:  Array of sample of x-type events; shape (N, 19, 3)
#         yEvents:  Array of sample of y-type events; shape (M, 19, 3)
#        OTSCHEME:  Determines what scheme will be used to calculate the OT distance.
#                   Note the exact meaning varies somewhat depending on the choice of COSTSCHEME.
#                   It's a dictionary of 3 booleans cooresponding to whether the PT is normalized,
#                   whether the OT calculation is balanced, and whether the zero padding should be
#                   removed. Namely,
#                   OTSCHEME['normPT']      ==True:  Means that the pT should be normalized;
#                                          ==False:  Means that the pT should be unnormalized
#                   OTSCHEME['balanced']    ==True:  Means that the OT calculation should be balanced;
#                                          ==False:  Means that the OT calculation should be unbalanced
#                   OTSCHEME['noZeroPad']   ==True:  Means that the zero padding should be removed;
#                                          ==False:  Means that the zero padding should be kept
#                   OTSCHEME['individualOT']==True:  Means that the OT calculation is done on each species separately;
#                                          ==False:  Means that the OT calculation is done ignoring species type
#      COSTSCHEME:  Determines what scheme will be used to calculate the ground cost matrix. Options are:
#                   - 1DpT: Ground space is pT only
#                   - 2D:   Ground space is 2D (eta,phi); note mass is pT
#                   - 3D:   Ground space is 3D (pT,eta,phi); note mass is uniform
#          kwargs:  Dictionary containing keyword arguments for unbalanced OT calculation; default is an empty dictionary
#          MATRIX:  Whether to return wXY as an (N,M) array or as a (N*M,) array

#   Outputs

#      wXY:  Matrix of the corresponding (squared) 2-Wasserstein distances, or list for default
#   """

#   #-- Sanity checks on inputs --#

#   # Check schemes
#   assert checkSCHEMES(OTSCHEME, COSTSCHEME)==True
#   if OTSCHEME['balanced']==False:
#     assert all([x in list(kwargs.keys()) for x in ['div', 'reg_m'] ])

#   N = xEvents.shape[0]
#   M = yEvents.shape[0]

#   #-- Preliminaries --#
#   # Create objects for outputs
#   wXY = np.zeros(shape=(N,M))

#   #-- Loop over pairs of events --#
#   # We use itertools to make looping more efficient (i.e. do double for loops)
#   dummyArr_1 = np.arange(N)
#   dummyArr_2 = np.arange(M)
#   wXY_list = []
#   for (i, j) in itertools.product(dummyArr_1, dummyArr_2):
#     #-- Get source and target data points--#
#     # Remove zero-padding if specified
#     if OTSCHEME['noZeroPad']==True:
#       sMask = ~np.all(xEvents[i, :, :] == 0., axis=1) # Mask to remove zero rows
#       tMask = ~np.all(yEvents[j, :, :] == 0., axis=1)

#       xs = xEvents[i, :, :][sMask] # "source data points"
#       xt = yEvents[j, :, :][tMask] # "target data points"
#     else:
#       xs = xEvents[i, :, :] # "source data points"
#       xt = yEvents[j, :, :] # "target data points"

#     #-- Get masses --#
#     a,b = getMasses(xs, xt, OTSCHEME, COSTSCHEME)

#     #-- Normalize pT coordinate if specified --#
#     #! Might be more efficient to do this before getMasses() but then will need to change getMasses()
#     if OTSCHEME['normPT']==True:
#       totPTs, totPTt = xs[:,0].sum(), xt[:,0].sum()
#       xs[:,0], xt[:,0] = xs[:,0]/totPTs, xt[:,0]/totPTt

#     #-- Get cost function and append to list --#
#     cxy = calcGroundCostMatrix(xs, xt, COSTSCHEME)

#     #-- Calculate the unbalanced Wasserstein distance --#
#     if OTSCHEME['balanced']==False:
#       wXY[i,j] = ot.unbalanced.mm_unbalanced2(a, b, cxy, reg_m=kwargs['reg_m'], div=kwargs['div'])
#       wXY_list.append(wXY[i,j])
#     else:
#       wXY[i,j] = ot.emd2(a, b, cxy)
#       wXY_list.append(wXY[i,j])

#   if Matrix == True:
#     return wXY

#   else:
#     return np.asarray(wXY_list)

def calcOTDistance_non_square(xEvents, yEvents):
  N = xEvents.shape[0]
  M = yEvents.shape[0]
  wXY = np.zeros(shape=(N,M))
  
  for i,j in tqdm(itertools.product(range(N), range(M)), total=N*M):
    _, wXY[i,j] = get_W2_Kin(xEvents[i], yEvents[j])
  
  return wXY

def parallel_OT_helper(events: np.ndarray, starting_index, events_per_thread):
    num_events = events.shape[0]
    
    result = np.zeros(shape=(events_per_thread, num_events))
    
    for i in range(events_per_thread):
        for j in range(starting_index+i):
          _, result[i,j] = get_W2_Kin(events[starting_index+i], events[j])
             
    return result

def parallel_OT_square(event_set, events_per_thread):
    num_events = len(event_set)
    assert num_events % events_per_thread == 0, 'Number of events must be divisible by the number of events per thread'
    num_threads = num_events // events_per_thread
    
    with cf.ProcessPoolExecutor() as executor:
        futures = [executor.submit(parallel_OT_helper, event_set, i*events_per_thread, events_per_thread) for i in range(num_threads)]

    results = [f.result() for f in futures]

    distance_matrix = np.vstack(results)
    distance_matrix = distance_matrix + distance_matrix.T

    return distance_matrix

def parallel_OT_non_square(event_set1, event_set2, events_per_thread):
    num_events = len(event_set1)
    assert num_events % events_per_thread == 0, 'Number of events in the first set must be divisible by the number of events per thread'
    num_threads = num_events // events_per_thread
    
    with cf.ProcessPoolExecutor() as executor:
        futures = [executor.submit(calcOTDistance_non_square, event_set1[events_per_thread*i:events_per_thread*(i+1)], event_set2) for i in range(num_threads)]

    results = [f.result() for f in futures]

    distance_matrix = np.vstack(results)

    return distance_matrix