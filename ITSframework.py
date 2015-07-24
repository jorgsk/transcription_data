"""
The ITS class holds sequence and experimental information about an ITS.

Associated functions (methods) calculate sequence-derived values, such as
translocation equilibria, or other such values.
"""
import numpy as np
import Energycalc as Ec
from ipdb import set_trace as debug  # NOQA

# So you can import optiom for calc Keq
import sys
sys.path.append('/home/jorgsk/Dropbox/phdproject/transcription_initiation/equilibrium/')


class ITS(object):

    def __init__(self, sequence, name='noname', PY=-1, PY_std=-1, APR=-1, msat=-1):
        # Set the initial important data.
        self.name = name
        # sequence is the non-template strand == the "same as RNA-strand"
        # ASSUMING 2006 DNA fragment here
        self.sequence = sequence + 'TAAATATGGC'
        self.PY = PY
        self.PY_std = PY_std
        self.APR = APR
        self.msat = int(msat)

        # N25 has a lot of extra data from additional experiments
        if self.name == 'N25':
            self.N25_specific_data()

        # Name of quantitations used to calculate AP, PY, etc.
        self.quantitations = []

        # Three ways of calculating abortive probability... find the diff
        self.abortiveProb_old = -1
        self.abortiveProb_std_old = -1  # std
        self.abortive_prob = -1  # first normalizing, then averaging, then AP
        self.abortive_prob_first_mean = -1  # first averaging signal, then AP

        # Raw quants
        self.rawAbortive = {}       # one entry for each quantitation

        # FL
        self.fullLength = {}

        # Mean fraction (abortive + FL) and raw data (abortive + FL)
        self.fraction = -1
        self.mean_raw_data = -1
        # Mean abortive fraction fraction[:-1] and re-normalized (/.sum())
        self.abortive_fraction = -1
        self.abortive_fraction_normal = -1

        # The SE sum of equilibrium constants (Keqs)
        self.SE = -1

        # make a dinucleotide list of the ITS sequence
        __seqlen = len(self.sequence)
        __indiv = list(self.sequence)
        __dinucs = [__indiv[c] + __indiv[c+1] for c in range(__seqlen-1)]

        self.dinucs = __dinucs

        self.rna_dna_di = [Ec.NNRD[di] for di in __dinucs]

        # Index 0, 1 etc is not relevant for translocation equilibrium because
        # there is no change in length
        self.dna_dna_di = [Ec.NNDD[di] for di in __dinucs]

        # Index 0 is the first translocation step
        self.dinucleotide_delta_g_f = [Ec.dinucleotide_deltaG_f[di] for di in __dinucs]
        self.dinucleotide_delta_g_b = [Ec.dinucleotide_deltaG_b[di] for di in __dinucs]

    def __repr__(self):
        return "{0}, PY: {1}".format(self.name, self.PY)

    def N25_specific_data(self):

        # From Vo 2003: pct yield of unproductive complexes only
        self.N25_unproductive_pct_yield = np.asarray([63.0, 16.0, 9.0, 2.2,
                                                      3.4, 1.5, 3.0, 0.9, 0.9,
                                                      0.0, 0.0, 0.0, 0.0, 0.0,
                                                      0.0, 0.0, 0.0, 0.0,
                                                      0.0])

        self.N25_unproductive_ap = calc_abortive_probability(self.N25_unproductive_pct_yield)

        # From Hsu 2006: pct yield when using GreB
        self.N25_GreB_pct_yield = np.asarray([60.4, 8.7, 7.9, 1.4, 0.9, 0.4,
                                              1.0, 0.5, 0.2, 0.1, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 18.5])

        self.abortive_prob_GreB = calc_abortive_probability(self.N25_GreB_pct_yield)

        # From Vo 2003:
        vo03_ab_pct = np.asarray([33.95, 12.73, 12.43, 9.36, 11.41, 3.95,
                                  10.09, 3.80, 1.76, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0])

        # What is the FL pct? Well, if the steady state reaction ran for 10
        # minutes, the FL pct is around 12% according to Fig 12. B.
        # So remove 12% from the aortive pct by weight and add to FL
        fl_pct = 12
        adjusted = vo03_ab_pct - fl_pct * vo03_ab_pct / 100.
        adjusted[-1] = fl_pct
        # But the 2nt pct and the FL pct is something you're changing later,
        # so perhaps it does not matter so much what you put here.

        self.vo03_fraction = adjusted / adjusted.sum()
        self.vo03_ap = calc_abortive_probability(self.vo03_fraction)

    def sane(self):
        """
        Verify that raw abortive data and full length data are in place
        """

        if self.rawAbortive == []:
            print('No raw data to work with!')
            return False

        elif self.fullLength == {}:
            print('No full length (FL) to work with!')
            return False

        else:
            return True

    def calc_PY(self):
        """
        Calculate PY (mean and std).
        """

        if not self.sane():
            return

        # store the PY for each quantitation
        self._PYraw = {}

        for quant in self.quantitations:

            totalRNA = sum(self.rawAbortive[quant]) + self.fullLength[quant]
            self._PYraw[quant] = self.fullLength[quant]/totalRNA

        self.PY = np.mean([py for py in self._PYraw.values()])
        self.PY_std = np.std([py for py in self._PYraw.values()])

    def calc_MSAT(self):
        """
        MSAT is read from Hsu. 2006 elsewhere, but we can calculate it from
        the % transcripts. Here, assume MSAT is the position after which 0.1%
        of RNA is left.
        """

        # Get pc of ONLY the abortive fraction
        abortive_only_pct = 100 * self.abortive_fraction_normal
        # Calculate the % RNA longer than i
        pct_longer_than = 100. - np.cumsum(abortive_only_pct)

        # Accept as MSAT the position where less than 0.1% of RNA is left
        limit = 0.1
        indx = np.where(pct_longer_than < limit)[0][0]

        # 2nt is index 0
        # This link between index and value is really bad. Should use pandas
        # from the beginning with proper index ... damn it
        self.msat_calculated = indx + 2

    def calc_AP_unproductive(self):
        """
        You got unproductive APs for N25 from Vo 2003.

        Here, the 2nt product is 1.75 times that of the normal simulation:
        but, the normal simulation has about 9% unproductive guys in them.
        Assuming this does not increase with time, you could probably subtract
        the unproductive abortive stuff from N25.

        Approach: calculate an elevated abortive % at +2 and spread the rest
        according to the abortive pattern; it's clearly correlated, just have
        a look at N25 in Vo.

        +3 is for N25 also elevated for nonproductive complexes in Vo, but
        that is second order for now.
        """

        # Start of with the 1.75 or 80% rule and move on from there. It's
        # about the principle, not the details.

        abortive_frac_after_2nt = self.abortive_fraction_normal[1:]

        # estimate non-productive abortive % for 2nt RNA (dinucleoside tetraphosphate?)
        nonprod_2nt_est = self.fraction[0] * 1.75
        nonprod_2nt_est = min(nonprod_2nt_est, 80.0)  # None above 80% (Vo 2003)
        nonprod_2nt_est = max(nonprod_2nt_est, 60.0)  # None below 60% (Vo 2003)

        # what remains to be spread over the rest
        remaining = 100. - nonprod_2nt_est
        the_rest = remaining * abortive_frac_after_2nt

        # nonproductive pct
        self.nonproductive_pct = np.asarray([nonprod_2nt_est] + the_rest.tolist())
        # So, 0.09 % of abortive RNA comes from the +18 position

        ap = calc_abortive_probability(self.nonproductive_pct)

        assert sum(ap > 1) == 0

        self.unproductive_ap = ap

    def calc_AP_GreB(self):
        """
        Use raw data from GreB+ experiments to calculate AP.
        """

    def calc_AP(self):
        """
        Calc AP using two different methods. One methods first normalizes each
        quantiation and then averages the normalized values. This is to take
        account for different total signal, but preserving the relative
        signal. I think that is the best method, really. The slight challenge
        is that we are averaging fractions that have different denominators;
        but, since the denominator is Total RNA, it does not matter.

        The other approach is to first average all signals and then calculate
        AP of the averaged signal. This is more correct in a sense, but if
        some quantitations show an important trend but has lower total signal,
        for example due to radioactive decay, then that trend will be lost.
        """

        normalized = {}
        raw_data = {}

        # go through each quantitation
        for quant in self.quantitations:

            # Read raw transcript data
            raw_abortive = np.asarray(self.rawAbortive[quant])
            raw_FL = self.fullLength[quant]
            # add together
            all_data = np.append(raw_abortive, raw_FL)
            assert np.isnan(all_data).sum() == 0
            # normalize
            normalized[quant] = all_data / all_data.sum()
            # Just add raw data
            raw_data[quant] = np.append(raw_abortive, raw_FL)

        # Calculate the respective means
        self.fraction = np.mean(normalized.values(), axis=0)
        self.fraction_std = np.std(normalized.values(), axis=0)
        # Also set the abortive fraction only, since this is used alot
        self.abortive_fraction = self.fraction[:-1]
        self.abortive_fraction_normal = self.abortive_fraction / self.abortive_fraction.sum()

        self.mean_raw_data = np.mean(raw_data.values(), axis=0)

        # calculate the two APs
        self.abortive_prob = calc_abortive_probability(self.fraction)
        self.abortive_prob_first_mean = calc_abortive_probability(self.mean_raw_data)

        assert sum([1 for ap in self.abortive_prob if ap < 0]) == 0
        assert sum([1 for ap in self.abortive_prob_first_mean if ap < 0]) == 0

    def calc_AP_old(self):
        """
        Use the raw data to calculate AP (mean and std).

        And then average the AP for the different quantitations. This seems
        like a wrong approach at first, but it has some merit. For example,
        the IQ raw data from each experiment may not be comparable. So if an
        important trend is present in a sample which has for some reason has
        lower IQ values, then this trend will not make a contribution if the
        raw IQ values are averaged.

        You need to compare the three methods: and perhaps stick with the one
        that produces the least variance?? :S Naah, that is not a good measure
        here.
        """

        if not self.sane():
            return

        # store the AP for each quantitation
        self._APraw = {}
        self.totAbort = {}

        # go through each quantitation
        for quant in self.quantitations:

            # Read raw transcript data
            raw_abortive = np.asarray(self.rawAbortive[quant])
            raw_FL = self.fullLength[quant]

            # add together
            all_data = np.append(raw_abortive, raw_FL)

            # calculate APs
            aps = calc_abortive_probability(all_data)

            # Do not save AP of FL
            self._APraw[quant] = aps[:-1]

        self.abortiveProb_old = np.nanmean(self._APraw.values(), axis=0)
        self.abortiveProb_std_old = np.nanstd(self._APraw.values(), axis=0)

        self.totAbortMean = np.nanmean(self.totAbort.values(), axis=0)

        # test, no abortive probability should be less than zero
        assert sum([1 for ap in self.abortiveProb_old if ap < 0]) == 0

    def calc_keq(self, c1, c2, c3, msat_normalization, rna_len):
        """
        Calculate Keq_i for each i in [2,rna_len]
        """

        # hard-coded constants hello Fates!
        RT = 1.9858775*(37 + 273.15)/1000  # divide by 1000 to get kcalories

        dna_dna = self.dna_dna_di[:rna_len]
        rna_dna = self.rna_dna_di[:rna_len-1]
        dg3d = self.dinucleotide_delta_g_b[:rna_len-1]

        # equilibrium constants at each position
        import optim

        self.keq = optim.keq_i(RT, rna_len, dg3d, dna_dna, rna_dna, c1, c2, c3)

        if msat_normalization:
            # keqs[0] is for a 2-mer RNA
            # keqs[1] is for a 3-mer RNA
            # ...
            # keqs[n-2] is for an (n)-mer RNA
            # so if msat = 11, there can be no equilibrium constant after index 9
            self.keq[self.msat-1:] = np.nan

    def calc_purines(self):
        """
        ATGCCA -> [101001]
        """
        self.purines = [1 if nuc in ['G', 'A'] else 0 for nuc in self.sequence[:20]]


def calc_abortive_probability(data):
    """
    data: numpy array of transcript product, either in absolute or relative
    numbers.

    AP: probability than transcript will be aborted at the ith position.

    Example:

    RNA Yields:
    Pos2: 40%
    Pos3: 10%
    Pos4: 30%
    Pos5: 20% (FL)

    So % RNAP is
    Pos2: 100%
    Pos3: 60%
    Pos4: 50%
    Pos5: 20% (FL)

    So AP becomes
    Pos2: 40/100 => 40%
    Pos3: 10/60  => 16%
    Pos2: 30/50  => 60%
    Pos2: 20/20  => 100% (all transcript "aborts" at FL)
    """

    # Work with np arrays
    data = np.asarray(data)

    # Normalize data (does not matter if it is already normalized)
    normdata = data / data.sum()

    # Find frac RNAP remaining up until position i
    frac_RNAP = np.asarray([normdata[i:].sum() for i in range(data.size)])

    # Round. so that 0.001 becomes 0; so that 0.99999 becomes 1
    frac_RNAP = np.round(frac_RNAP, 6)

    # Calc the APs
    ap = normdata / frac_RNAP

    # nan likely means 0/0
    ap[np.isnan(ap)] = 0

    # We don't need no digits. Edit ...eeeh yes, you'll lose %ages that way.
    # Ehm, nope, seems occasionally you get an AP of 1.0003 or 0.000001,
    # which really should be 1 or 0.
    ap = np.round(ap, 2)

    return ap


if __name__ == '__main__':

    test_its = ITS('GATTACAGATTACAGATTACA', name='test_sequence')
    test_its.forward_transclocation(method='hein_et_al')

