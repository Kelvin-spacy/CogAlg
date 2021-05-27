'''
line_PPs is a 2nd-level 1D algorithm, its input is Ps formed by the 1st-level line_patterns.
It cross-comPares Ps (s, L, I, D, M, dert_, layers) and evaluates them for deeper cross-comParison.
Range or derivation of cross-comP is selectively increased if the match from prior-order cross-comP is above threshold:
comP s: if same-sign,
        cross-sign comP is borrow, also default L and M (core param) comP?
        discontinuous comP up to max rel distance +|- contrast borrow, with bi-directional selection?
    comP (L, I, D, M): equal-weight, select redundant I | (D,M),  div L if V_var * D_vars, and same-sign d_vars?
        comP (dert_):  lower comPosition than layers, if any
    comP (layers):  same-derivation elements
        comP (P_):  sub patterns
Increment of 2nd level alg over 1st level alg should be made recursive, forming relative-level meta-algorithm.
ComParison distance is extended to first match or maximal accumulated miss over comPared derPs, measured by roL*roM?
Match or miss may be between Ps of either sign, but comParison of lower P layers is conditional on higher-layer match
ComParison between two Ps is of variable-depth P hierarchy, with sign at the top, until max higher-layer miss.
This is vertical induction: results of higher-layer comParison predict results of next-layer comParison,
similar to lateral induction: variable-range comParison among Ps, until first match or max prior-Ps miss.
Resulting PPs will be more like 1D graphs, with explicit distances between nearest element Ps.
This is different from 1st level connectivity clustering, where all distances between nearest elements = 1.
'''

import numpy as np
from line_patterns import *
from class_cluster import ClusterStructure, NoneType

class CderP(CP):
    smP = NoneType
    P = object
    neg_M = int
    neg_L = int
    mP = NoneType


class CPP(CderP): 
    P_ = list
    sub_layers = list

ave = 100  # ave dI -> mI, * coef / var type
'''
no ave_mP: deviation comPuted via rM  # ave_mP = ave*3: comP cost, or n vars per P: rep cost?
'''
ave_div = 50
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?
ave_M = 100  # search stop
ave_Ls = 3

def comP_P_(P_):  # cross-comPare patterns within horizontal line

    derP_ = []  # comP_P_ forms array of alternating-sign derPs (derivatives + P): output of pair-wise comP_P

    for i, P in enumerate(P_):
        neg_M = vmP = smP = _smP = neg_L = 0  # initialization
        M = P.M
        for j, _P in enumerate(P_[i + 1:]):  # variable-range comP, no last-P displacement, just shifting first _P
            if  M + neg_M > ave_M:  # search while net_M > ave or 1st _P, no selection by M sign

                derP, _L, _smP = comP_P(P, _P, neg_M, neg_L)
                smP, vmP, neg_M, neg_L, P = derP.smP, derP.mP, derP.neg_M, derP.neg_L, derP.P
                if smP:
                    P_[i + 1 + j].smP = True  # backward match per P: __smP = True
                    derP_.append(derP)
                    break  # nearest-neighbour search is terminated by first match
                else:
                    neg_M += vmP  # accumulate contiguous miss: negative mP
                    neg_L += _L   # accumulate distance to match
                    if j == len(P_):
                        # last P is a singleton derP, derivatives are ignored:
                        derP_.append(CderP(mP=vmP, neg_M=neg_M, neg_L=neg_L,smP=smP or _smP, P=P ))
                    '''                     
                    no contrast value in neg derPs and PPs: initial opposite-sign P miss is expected
                    neg_derP derivatives are not significant; neg_M obviates distance * decay_rate * M '''
            else:
                derP_.append(CderP(mP=vmP, neg_M=neg_M, neg_L=neg_L,smP=smP or _smP, P=P))
                # smP is ORed bilaterally, negative for singleton derPs only
                break  # neg net_M: stop search

    return derP_


def comP_P(P, _P, neg_M, neg_L):  # multi-variate cross-comP, _smP = 0 in line_patterns

    #sign, L, I, D, M, dert_, sub_H, _smP = P.sign, P.L, P.I, P.D, P.M, P.dert_, P.sub_layers, P.smP
    #_sign, _L, _I, _D, _M, _dert_, _sub_H, __smP = _P.sign, _P.L, _P.I, _P.D, _P.M, _P.dert_, _P.sub_layers, _P.smP
    derP = CderP()
    sign, L, _smP= P.sign, P.L, P.smP
    _sign, _L = _P.sign, P.L
    dC_ave = ave_M * ave_rM ** (1 + neg_L / L)  # average match projected at current distance: neg_L, add coef / var?
    # if param fderived: m = min(var,_var) - dC_ave,
    # else:              m = dC_ave - abs(d_var), always a deviation:
    # append base param at index 0
    derP.L = [P.L,min(L, _L) - dC_ave,L-_L]
    derP.I = [P.I,dC_ave - abs(dI),P.I - _P.I]
    derP.M = [P.M,min(P.M, _P.M) - dC_ave,P.M - _P.M]
    derP.D = [P.D,min(P.D, _P.D) - dC_ave,P.D - _P.D]
    
    
    derP.mP = derP.I[1] + derP.L[1]  + derP.M[1] + derP.D[1]  # match(P, _P), no I: overlap, for regression to 0der-representation?
    if sign == _sign: mP *= 2  # sign is MSB, value of sign match = full magnitude match?

    smP = derP.mP > 0
    
    if smP:  # positive forward match, comPare sub_layers between P.sub_H and _P.sub_H:
        dert_sub_H = []  # sub hierarchy, abbreviation for new sub_layers

        if P.sub_layers and _P.sub_layers:  # not emPty sub layers
            for sub_P, _sub_P in zip(P.sub_layers, _P.sub_layers):

                if P and _P:  # both forks exist
                    Ls, fdP, fid, rdn, rng, sub_P_ = sub_P[0]
                    _Ls, _fdP, _fid, _rdn, _rng, _sub_P_ = _sub_P[0]
                    # fork comParison:
                    if fdP == _fdP and rng == _rng and min(Ls, _Ls) > ave_Ls:
                        dert_sub_P_ = []
                        sub_mP = 0
                        # comPare all sub_Ps to each _sub_P, form dert_sub_P per comPared pair
                        for sub_P in sub_P_:  # note name recycling in nested loop
                            for _sub_P in _sub_P_:
                                dert_sub_P, _, _ = comP_P(sub_P, _sub_P, neg_M=0, neg_L=0)  # ignore _sub_L, _sub_smP?
                                sub_mP += dert_sub_P.mP  # sum sub_vmPs in derP_layer
                                dert_sub_P_.append(dert_sub_P)

                        dert_sub_H.append((fdP, fid, rdn, rng, dert_sub_P_))  # add only layers that have been comPared
                        mP += sub_mP  # of comPared H, no specific mP?
                        if sub_mP < ave_M:
                            # potentially mH: trans-layer induction?
                            break  # low vertical induction, deeper sub_layers are not comPared
                    else:
                        break  # deeper P and _P sub_layers are from different intra_comP forks, not comParable?
    derP.smP = smP
    derP.P = P
    derP.neg_M = neg_M
    derP.neg_L = neg_L
    return derP, _L, _smP


def form_PPm(derP_):  # cluster derPs into PPm s by mP sign, eval for div_comP per PPm

    PPm_ = []
    sub_layers = []
    #_smP, mP, neg_M, neg_L, _P, ML, DL, MI, DI, MD, DD, MM, DM = \
    #derP.smP, derP.mP, derP.neg_M, derP.neg_L, derP.P, derP.ML, derP.DL, derP.MI, derP.DI, derP.MD, derP.DD, derP.MM, derP.DM
    

    for i, derP in enumerate(derP_, start=1):
        if i == 0:
            _smP = derP.smP
            _P = derP.P
            P_ = [_P]
            _derP = derP
            continue   
        smP = derP.smP
        if smP != _smP:
            # terminate PPm:
            _derP.smP = smP     # previous _derP.smP equals current smP
            _derP.P = P_        # previous _derP.P equals P_ array
            PPm_.append(CPP(L=_derP.L, I=_derP.I, D=_derP.D, M=_derP.M, mP=_derP.mP, neg_M=_derP.net_M, neg_L=_derP.neg_L, smP=_derP.smP, P_=_derP.P, sub_layers=sub_layers))
            # initialize PPm with current derP:

            #_smP, mP, neg_M, neg_L, _P, ML, DL, MI, DI, MD, DD, MM, DM = \
            #derP.smP, derP.mP, derP.neg_M, derP.neg_L, derP.P, derP.ML, derP.DL, derP.MI, derP.DI, derP.MD, derP.DD, derP.MM, derP.DM
            _smP = dert_P.smP  # Current smP becomes Previous smP i-e _smP
            _P = derP.P      #Current P becomes previous P i-e _P
            P_ = [_P]
            _derP = derP
        else:
            # accumulate PPm with current derP:
            _derP.accum_from(derP) #accum numerical params
            der = _derP.accum_list(derP,excluded=('P_','smP')) #accum list params
            _derP.L = der['L']
            _derP.I = der['I']
            _derP.D = der['D']
            _derP.M = der['M']
            P_.append(derP.P)
        _smP = smP
    # pack last PP:
    PPm_.append(CPP(L=_derP.L, I=_derP.I, D=_derP.D, M=_derP.M, mP=_derP.mP, neg_M=_derP.net_M, neg_L=_derP.neg_L, smP=smP, P_=P_, sub_layers=sub_layers))
    #P_l = [len(P.P_) for P in PPm_]
    #ave_PPm_P = sum(P_l)/len(P_l)
    if len(PPm_.P_) > 8:
        intra_PPm(PPm_,fid=False, rdn=1, rng=3)

    return PPm_


def intra_PPm(PPm_,fid,rdn,rng):
    comb_layers = []
    rdert_ = []
    sub_PPm_ = []
    
    for PP in PPm_:
        if PP.smP:
            rdert_ = range_comP(PP.P_)
            sub_PPm_ = form_PPm(rdert)
            Ls = len(sub_PPm_)
            PP.sub_layers += [[(Ls, fid, rdn, rng, sub_PPm_)]]
            if len(sub_PPm_) > 8:
                PP.sub_layers += intra_PPm_(sub_PPm_, fid, rdn + 1 + 1 / Ls, rng * 2 + 1)
                comb_layers = [comb_layers + sub_layers for comb_layers, sub_layers in
                               zip_longest(comb_layers, PP.sub_layers, fillvalue=[])]





def range_comP(P_):
    rdert_ = []
    for n in range(0,len(P_)):
        rdert.append(comP_P_(P_[n]))

    return rdert

            






''' 
    Each PP is evaluated for intra-processing, not per P: results must be comParable between consecutive Ps): 
    - incremental range and derivation as in line_patterns intra_P, but over multiple params, 
    - x param div_comP: if internal comPression: rm * D * L, * external comPression: PP.L * L-proportional coef? 
    - form_par_P if param Match | x_param Contrast: diff (D_param, ave_D_alt_params: co-derived co-vary? neg val per P, else delete?
    
    form_PPd: dP = dL + dM + dD  # -> directional PPd, equal-weight params, no rdn?  
    if comP I -> dI ~ combined d_derivatives, then project ave_d?
    
    L is summed sign: val = S val, core ! comb?  
'''




def div_comP_P(PP_):  # draft, check all PPs for x-param comP by division between element Ps
    '''
    div x param if projected div match: comPression per PP, no internal range for ind eval.
    ~ (L*D + L*M) * rm: L=min, positive if same-sign L & S, proportional to both but includes fractional miss
    + PPm' DL * DS: xP difference comPression, additive to x param (intra) comPression: S / L -> comP rS
    also + ML * MS: redundant unless min or converted?
    vs. norm param: Var*rL-> comP norm param, simPler but diffs are not L-proportional?
    '''
    for PP in PP_:
        if PP.M / (PP.L + PP.I + abs(PP.D) + abs(PP.dM)) * (abs(PP.dL) + abs(PP.dI) + abs(PP.dD) + abs(PP.dM)) > ave_div:
            # if irM * D_vars: match rate projects der and div match,
            # div if scale invariance: comP x dVars, signed
            ''' 
            | abs(dL + dI + dD + dM): div value ~= L, Vars correlation: stability of density, opposite signs cancel-out?
            | div_comP value is match: min(dL, dI, dD, dM) * 4, | sum of pairwise mins?
            '''
            _derP = PP.derP_[0]
            # smP, vmP, neg_M, neg_L, iP, mL, dL, mI, dI, mD, dD, mM, dM = P,
            _sign, _L, _I, _D, _M, _dert_, _sub_H, __smP = _derP[4]

            for i, derP in enumerate(PP.derP_[1:]):
                sign, L, I, D, M, dert_, sub_H, _smP = derP[4]
                # DIV comP L, SUB comP (summed param * rL) -> scale-independent d, neg if cross-sign:
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
                _sign = sign, _L = L, _I = I, _D = D, _M = M, _dert_ = dert_, _sub_H = sub_H, __smP = _smP
                '''
                m and d from comP_rate is more accurate than comP_norm?
                rm, rd: rate value is relative? 
                
                also define Pd, if strongly directional? 
                   
                if dP > ndP: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
                else:        dPP_rdn = 1; ndPP_rdn = 0
                '''
    return PP_


''' non-class version: 
def accum_PP(PP: dict, **params) -> None:
    PP.update({param: PP[param] + value for param, value in params.items()})
def comP_P_(P_):  # cross-comPare patterns within horizontal line
    derP_ = []  # comP_P_ forms array of alternating-sign derPs (derivatives + P): output of pair-wise comP_P
    for i, P in enumerate(P_):
        neg_M = vmP = smP = _smP = neg_L = 0  # initialization
        M = P[4]
        for j, _P in enumerate(P_[i+1 :]):  # variable-range comP, no last-P displacement, just shifting first _P
            if M - neg_M > ave_net_M:
                # search while net_M > ave, True for 1st _P, no select by M sign
                derP, _L, _smP = comP_P(P, _P, neg_M, neg_L)
                smP, vmP, neg_M, neg_L, P = derP[:5]
                if smP:
                    P_[i + 1 + j][-1] = True  # backward match per P: __smP = True
                    derP_.append(derP)
                    break  # nearest-neighbour search is terminated by first match
                else:
                    neg_M += vmP  # accumulate contiguous miss: negative mP
                    neg_L += _L   # accumulate distance to match
                    if j == len(P_):  # last P
                        derP_.append((smP or _smP, vmP, neg_M, neg_L, P, 0, 0, 0, 0, 0, 0, 0, 0))          
                    # no contrast value in neg derPs and PPs: initial opposite-sign P miss is expected
                    # neg_derP derivatives are not significant; neg_M obviates distance * decay_rate * M 
            else:
                derP_.append((smP or _smP, vmP, neg_M, neg_L, P, 0, 0, 0, 0, 0, 0, 0, 0))
                # smP is ORed bilaterally, negative for single (weak) derPs
                break  # neg net_M: stop search
    return derP_
def comP_P(P, _P, neg_M, neg_L):
    sign, L, I, D, M, dert_, sub_H, _smP = P  # _smP = 0 in line_patterns, M: deviation even if min
    _sign, _L, _I, _D, _M, _dert_, _sub_H, __smP = _P
    dL = L - _L
    mL = min(L, _L)  # - ave_rM * L?  L: positions / sign, derived: magnitude-proportional value
    dI = I - _I  # proportional to distance, not I?
    mI = ave_dI - abs(dI)  # I is not derived, match is inverse deviation of miss
    dD = D - _D  # sum if opposite-sign
    mD = min(D, _D)  # - ave_rM * D?  same-sign D in dP?
    dM = M - _M  # sum if opposite-sign
    mM = min(M, _M)  # - ave_rM * M?  negative if x-sign, M += adj_M + deep_M: P value before layer value?
    mP = mL + mM + mD  # match(P, _P) for derived vars, mI is already a deviation
    proj_mP = (L + M + D) * (ave_rM ** (1 + neg_L / L))  # projected mP at current relative distance
    vmP = mI + (mP - proj_mP)  # deviation from projected mP, ~ I*rM contrast value, +|-? replaces mP?
    smP = vmP > 0
    if smP:  # forward match, comPare sub_layers between P.sub_H and _P.sub_H (sub_hierarchies):
        dert_sub_H = []
        if P[6] and _P[6]: # not emPty sub layers
            for (Ls, fdP, fid, rdn, rng, sub_P_), (_Ls, _fdP, _fid, _rdn, _rng, _sub_P_) in zip(*P[6], *_P[6]):
                # fork comParison:
                if fdP == _fdP and rng == _rng and min(Ls, _Ls) > ave_Ls:
                    dert_sub_P_ = []
                    sub_mP = 0
                    # comPare all sub_Ps to each _sub_P, form dert_sub_P per comPared pair
                    for sub_P in sub_P_:
                        for _sub_P in _sub_P_:
                            dert_sub_P, _, _ = comP_P(sub_P, _sub_P, neg_M=0, neg_L=0)  # ignore _sub_L, _sub_smP?
                            sub_mP += dert_sub_P[1]  # sum sub_vmPs in derP_layer
                            dert_sub_P_.append(dert_sub_P)
                    dert_sub_H.append((fdP, fid, rdn, rng, dert_sub_P_))  # only layers that have been comPared
                    vmP += sub_mP  # of comPared H, no specific mP?
                    if sub_mP < ave_net_M:
                        # or mH: trans-layer induction?
                        break  # low vertical induction, deeper sub_layers are not comPared
                else:
                    break  # deeper P and _P sub_layers are from different intra_comP forks, not comParable?
    return (smP, vmP, neg_M, neg_L, P, mL, dL, mI, dI, mD, dD, mM, dM), _L, _smP
def form_PPm(derP_):  # cluster derPs by mP sign, positive only: no contrast in overlapping comP?
    PPm_ = []
    # initialize PPm with first derP:
    _smP, mP, neg_M, neg_L, _P, ML, DL, MI, DI, MD, DD, MM, DM = derP_[0]  # positive only, no contrast?
    P_ = [_P]
    for i, derP in enumerate(derP_, start=1):
        smP = derP[0]
        if smP != _smP:
            PPm_.append([_smP, mP, neg_M, neg_L, P_, ML, DL, MI, DI, MD, DD, MM, DM])
            # initialize PPm with current derP:
            _smP, mP, neg_M, neg_L, _P, ML, DL, MI, DI, MD, DD, MM, DM = derP
            P_ = [_P]
        else:
            # accumulate PPm with current derP:
            smP, mP, neg_M, neg_L, P, mL, dL, mI, dI, mD, dD, mM, dM = derP
            mP+=mP; neg_M+=neg_M; neg_L+=neg_L; ML+=mL; DL+=dL; MI+=mI; DI+=dI; MD+=mD; DD+=dD; MM+=mM; DM+=dM
            P_.append(P)
        _smP = smP
    PPm_.append([_smP, mP, neg_M, neg_L, P_, ML, DL, MI, DI, MD, DD, MM, DM])  # pack last PP
    return PPm_
    # in form_PPd:
    # dP = dL + dM + dD  # -> directional PPd, equal-weight params, no rdn?
    # ds = 1 if Pd > 0 else 0
def accum_PP(PP: dict, **params) -> None:
    PP.update({param: PP[param] + value for param, value in params.items()}) 
'''