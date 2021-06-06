'''
line_PPs is a 2nd-level 1D algorithm, its input is Ps formed by the 1st-level line_patterns.
It cross-compares Ps (s, L, I, D, M, dert_, layers) and evaluates them for deeper cross-comparison.
Range or derivation of cross-comp is selectively increased if the match from prior-order cross-comp is above threshold,
search up to max rel distance +|- contrast borrow, with bi-directional selection?
-
comp s: if same-sign, cross-sign comp is borrow, also default L and M (core param) comp?
comp (L, I, D, M): equal-weight, select redundant I | (D,M),  div L if V_var * D_vars, and same-sign d_vars?
comp (dert_):  lower composition than layers, if any
comp (layers):  same-derivation elements
comp (P_):  sub patterns
-
Increment of 2nd level alg over 1st level alg should be made recursive, forming relative-level meta-algorithm.
Comparison distance is extended to first match or maximal accumulated miss over compared derPs, measured by roL*roM?
Match or miss may be between Ps of either sign, but comparison of lower P layers is conditional on higher-layer match
-
Comparison between two Ps is of variable-depth P hierarchy, with sign at the top, until max higher-layer miss.
This is vertical induction: results of higher-layer comparison predict results of next-layer comparison,
similar to lateral induction: variable-range comparison among Ps, until first match or max prior-Ps miss.
-
Resulting PPs will be more like 1D graphs, with explicit distances between nearest element Ps.
This is different from 1st level connectivity clustering, where all distances between nearest elements = 1.
Negative PPms and PPds are not formed: contrast value can't be assumed because all Ps are terminated.
Still, form PPd for strong Pm s?
also deeper backward search for strong Pm s?
'''

import numpy as np
from line_patterns import *
from class_cluster import ClusterStructure, NoneType

class CderP(ClusterStructure):
    sign = bool
    mP = int
    dP = int
    neg_M = int
    neg_L = int
    P = list
    layer1 = list
    layer1_param = list
class Clayer(ClusterStructure):
    layer0 = list
    layer1 = list

class CPP(ClusterStructure):
    sign = bool
    mP = int
    dP = int
    neg_M = int
    neg_L = int
    P_ = list
    layers = list
    layers_param = list
    sub_layers = list

ave = 100  # ave dI -> mI, * coef / var type
# no ave_mP: deviation computed via rM  # ave_mP = ave* n_comp_params: comp cost, or n vars per P: rep cost?
ave_div = 50
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?
ave_M = 100  # search stop
ave_sub_M = 50  # sub_H comp filter
ave_Ls = 3
ave_PPM = 200


def search(P_):  # cross-compare patterns within horizontal line

    derP_ = []  # search forms array of alternating-sign derPs (derivatives + P): output of pair-wise comp_P
    der_P_ = []
    for i, P in enumerate(P_):
        neg_M = vmP = sign = _sign = neg_L = 0  # initialization

        for j, _P in enumerate(P_[i + 1:]):  # variable-range comp, no last-P displacement, just shifting first _P
            #if _P.M < P.M:      #check if P is already compared
            #    P._P_.append([_P,P])  #store consecutive Ps
            if P.M + neg_M > 0:  # search while net_M > ave_M * nparams or 1st _P, no selection by M sign
               # P.M decay with distance: * ave_rM ** (1 + neg_L / P.L): only for abs P.M?

                derP, _L, _sign = comp_P(P, _P, neg_M, neg_L)
                sign, vmP, neg_M, neg_L, P = derP.sign, derP.mP, derP.neg_M, derP.neg_L, derP.P
                if sign:
                    P_[i + 1 + j].sign = True  # backward match per P: __sign = True
                    derP_.append(derP)
                    break  # nearest-neighbour search is terminated by first match
                else:
                    neg_M += vmP  # accumulate contiguous miss: negative mP
                    neg_L += _L   # accumulate distance to match
                    if j == len(P_):
                        # last P is a singleton derP, derivatives are ignored:
                        derP_.append(CderP(sign=sign or _sign, mP=vmP, neg_M=neg_M, neg_L=neg_L, P=P ))
                    '''                     
                    no contrast value in neg derPs and PPs: initial opposite-sign P miss is expected
                    neg_derP derivatives are not significant; neg_M obviates distance * decay_rate * M '''
            else:
                derP_.append(CderP(sign=sign or _sign, mP=vmP, neg_M=neg_M, neg_L=neg_L, P=P))
                # sign is ORed bilaterally, negative for singleton derPs only
                break  # neg net_M: stop search

            if derP_:
                for derP in derP_:
                    if not derP.sign:  # check false sign
                        print('False sign in line' + str(y))

            PPm_ = form_PPm_(derP_)  # cluster derPs into PPms by the sign of mP
        for x0 in range(len(derP_)):
            if x >=2 and len(derP_)>3:  # check 
                _P_.append([derP_[x0-2].P,derP_[x0].P])  #set of P that has not been compared (__P,P)

    

    return PPm_


def comp_P(P, _P, neg_M, neg_L):  # multi-variate cross-comp, _sign = 0 in line_patterns

    dC_ave = ave_M * ave_rM ** (1 + neg_L / P.L)  # average match projected at current distance: neg_L, add coef / var?
    layer1,param_name = P.layer0.comp_param(_P.layer0, ave=dC_ave)  # comp_param may need to be edited

    mP = layer1[0][1] + layer1[1][1] + layer1[2][1] + layer1[3][1]  # match(P, _P), no I: for regression to 0der only?
    dP = layer1[0][0] + layer1[1][0] + layer1[2][2] + layer1[3][0]  # difference(P,_P)
    # each m is a deviation, better absolute?
    if P.sign == _P.sign: mP *= 2  # sign is MSB, value of sign match = full magnitude match?
    x0 = 'compared'
    sign = mP > 0
    if sign:  # positive forward match, compare sub_layers between P.sub_H and _P.sub_H:
        der_sub_H = []  # sub hierarchy, abbreviation for new sub_layers

        if P.sub_layers and _P.sub_layers:  # not empty sub layers
            for sub_P, _sub_P in zip(P.sub_layers, _P.sub_layers):

                if P and _P:  # both forks exist
                    Ls, fdP, fid, rdn, rng, sub_P_ = sub_P[0]
                    _Ls, _fdP, _fid, _rdn, _rng, _sub_P_ = _sub_P[0]
                    # fork comparison:
                    if fdP == _fdP and rng == _rng and min(Ls, _Ls) > ave_Ls:
                        dert_sub_P_ = []
                        sub_mP = 0
                        # compare all sub_Ps to each _sub_P, form dert_sub_P per compared pair
                        for sub_P in sub_P_:  # note name recycling in nested loop
                            for _sub_P in _sub_P_:
                                der_sub_P, _, _ = comp_P(sub_P, _sub_P, neg_M=0, neg_L=0)  # ignore _sub_L, _sub_sign?
                                sub_mP += der_sub_P.mP  # sum sub_vmPs in derP_layer
                                dert_sub_P_.append(der_sub_P)

                        der_sub_H.append((fdP, fid, rdn, rng, dert_sub_P_))  # add only layers that have been compared
                        mP += sub_mP  # of compared H, no specific mP?
                        if sub_mP < ave_sub_M:
                            # potentially mH: trans-layer induction?
                            break  # low vertical induction, deeper sub_layers are not compared
                    else:
                        break  # deeper P and _P sub_layers are from different intra_comp forks, not comparable?

    derP = CderP(sign=sign, mP=mP, dP=dP, neg_M=neg_M, neg_L=neg_L, P = P, layer1=layer1,layer1_param=param_name)

    return derP, _P.L, _P.sign



def form_PPm(derP_):  # cluster derPs into PPm s by mP sign, eval for div_comp per PPm

    PPm_ = []
    _derP = derP_[0]

    _sign = _derP.sign
    _P = _derP.P
    P_ = [_P]
    _layers = Clayer(layer0=_P.layer0,layer1=_derP.layer1)


    for i, derP in enumerate(derP_, start=1):  
        sign = derP.sign
        if sign != _sign:
            # terminate PPm:
            PPm_.append(CPP(sign=sign,mP=_derP.mP,neg_M=_derP.neg_M,neg_L=_derP.neg_L,sub_layers=sub_layers,P_=P_,layers=_layers))
            # initialize PPm with current derP:
            _sign = derP.sign
            _P = derP.P
            P_ = [_P]
            _derP = derP
            _layers = Clayer()
            
        else:
            # accumulate PPm with current derP:
            _derP.accum_from(derP)
            for i,param in enumerate(_derP.P.layer0):
                _layers.layer0[i]+=param
            for i, (d,m) in enumerate(derP.layer1):
                _layers.layer1[i][0] += d  
                _layers.layer1[i][1] += m
            
            P_.append(derP.P)
        _sign = sign
    # pack last PP:
    
    PPm_.append(CPP(sign=_sign,mP=_derP.mP,neg_M=_derP.neg_M,neg_L=_derP.neg_L,sub_layers=sub_layers,P_=P_,layers=_layers))

    return PPm_


def intra_PPm(PPm_,fid,rdn,rng):
    comb_layers = []
    rdert_ = []
    sub_PPm_ = []
    
    for PP in PPm_:
        if PP.smP:
            rdert_ = range_comp(PP.P_,PP.layer0[3])
            sub_PPm_ = form_PPm(rdert)
            Ls = len(sub_PPm_)
            PP.sub_layers += [[(Ls, fid, rdn, rng, sub_PPm_)]]
            if len(sub_PPm_) > 8:
                PP.sub_layers += intra_PPm_(sub_PPm_, fid, rdn + 1 + 1 / Ls, rng * 2 + 1)
                comb_layers = [comb_layers + sub_layers for comb_layers, sub_layers in
                               zip_longest(comb_layers, PP.sub_layers, fillvalue=[])]





def rng_search_P(P_,M):
    rderP_ = []
    ave_M = ave_M + (M / len(P_) / 2 #global ave_M + local ave_M
    for i,P in enumerate(P_):
        neg_M = vmP = sign = _sign = neg_L = 0  # not sure if this is needed here

        for j, _P in enumerate(P_[i + 1:]):  # variable-range comp, no last-P displacement, just shifting first _P
            if P.layer0[3] * (neg_L/P.layer0[0] * ave_rM) + neg_M > ave_M. :  # search while net_M > ave_M i-e sum of local and global match
                if not _P.layer0[3] > P.layer0[3]:   # skip previously compared P
                    rderP, _L, _sign = comp_P(P, _P, neg_M, neg_L)
                    sign, vmP, neg_M, neg_L, P = rderP.sign, rderP.mP, rderP.neg_M, rderP.neg_L, rderP.P
                    if sign:
                        P_[i + 1 + j].sign = True  # backward match per P: __sign = True
                        rderP_.append(rderP)
                        break  # nearest-neighbour search is terminated by first match
                    else:
                        neg_M += vmP  # accumulate contiguous miss: negative mP
                        neg_L += _L   # accumulate distance to match
                        if j == len(P_):
                            # last P is a singleton derP, derivatives are ignored:
                            rderP_.append(CderP(sign=sign or _sign, mP=vmP, neg_M=neg_M, neg_L=neg_L, P=P ))
                        
            else:
                rderP_.append(CderP(sign=sign or _sign, mP=vmP, neg_M=neg_M, neg_L=neg_L, P=P))
                # sign is ORed bilaterally, negative for singleton derPs only
                break  # neg net_M: stop search

    return rdert

            






''' 
    Each PP is evaluated for intra-processing, not per P: results must be comparable between consecutive Ps): 
    - incremental range and derivation as in line_patterns intra_P, but over multiple params, 
    - x param div_comp: if internal compression: rm * D * L, * external compression: PP.L * L-proportional coef? 
    - form_par_P if param Match | x_param Contrast: diff (D_param, ave_D_alt_params: co-derived co-vary? neg val per P, else delete?
    
    form_PPd: dP = dL + dM + dD  # -> directional PPd, equal-weight params, no rdn?  
    if comp I -> dI ~ combined d_derivatives, then project ave_d?
    
    L is summed sign: val = S val, core ! comb?  
'''

def div_comp_P(PP_):  # draft, check all PPs for x-param comp by division between element Ps
    '''
    div x param if projected div match: compression per PP, no internal range for ind eval.
    ~ (L*D + L*M) * rm: L=min, positive if same-sign L & S, proportional to both but includes fractional miss
    + PPm' DL * DS: xP difference compression, additive to x param (intra) compression: S / L -> comp rS
    also + ML * MS: redundant unless min or converted?
    vs. norm param: Var*rL-> comp norm param, simpler but diffs are not L-proportional?
    '''
    for PP in PP_:
        if PP.M / (PP.L + PP.I + abs(PP.D) + abs(PP.dM)) * (abs(PP.dL) + abs(PP.dI) + abs(PP.dD) + abs(PP.dM)) > ave_div:
            # if irM * D_vars: match rate projects der and div match,
            # div if scale invariance: comp x dVars, signed
            ''' 
            | abs(dL + dI + dD + dM): div value ~= L, Vars correlation: stability of density, opposite signs cancel-out?
            | div_comp value is match: min(dL, dI, dD, dM) * 4, | sum of pairwise mins?
            '''
            _derP = PP.derP_[0]
            # sign, vmP, neg_M, neg_L, iP, mL, dL, mI, dI, mD, dD, mM, dM = P,
            _sign, _L, _I, _D, _M, _dert_, _sub_H, __sign = _derP[4]

            for i, derP in enumerate(PP.derP_[1:]):
                sign, L, I, D, M, dert_, sub_H, _sign = derP[4]
                # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
                rL = L / _L
                # mL = whole_rL * min_L?
                dI = I * rL - _I  # rL-normalized dI, vs. nI = dI * rL or aI = I / L
                mI = ave - abs(dI)  # I is not derived, match is inverse deviation of miss
                dD = D * rL - _D  # sum if opposite-sign
                mD = min(D, _D)   # same-sign D in dP?
                dM = M * rL - _M  # sum if opposite-sign
                mM = min(M, _M)   # - ave_rM * M?  negative if x-sign, M += adj_M + deep_M: P value before layer value?

                mP = mI + mM + mD  # match(P, _P) for derived vars, defines norm_PPm, no ndx: single, but nmx is summed
                if mP > derP[1]:
                    rrdn = 1  # added to rdn, or diff alt, olp, div rdn?
                else:
                    rrdn = 2
                if mP > ave * 3 * rrdn:
                    rvars = mP, mI, mD, mM, dI, dD, dM  # redundant vars: dPP_rdn, ndPP_rdn, assigned in each fork?
                else:
                    rvars = []
                # append rrdn and ratio variables to current derP:
                PP.derP_[i] += [rrdn, rvars]
                # P vars -> _P vars:
                _sign = sign, _L = L, _I = I, _D = D, _M = M, _dert_ = dert_, _sub_H = sub_H, __sign = _sign
                '''
                m and d from comp_rate is more accurate than comp_norm?
                rm, rd: rate value is relative? 
                
                also define Pd, if strongly directional? 
                   
                if dP > ndP: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
                else:        dPP_rdn = 1; ndPP_rdn = 0
                '''
    return PP_