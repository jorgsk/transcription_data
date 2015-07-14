"""
The ITS class holds sequence and experimental information about an ITS.

Associated functions (methods) calculate sequence-derived values, such as
translocation equilibria, or other such values.
"""
import numpy as np
import Energycalc as Ec
from ipdb import set_trace as debug  # NOQA


class ITS(object):

    def __init__(self, sequence, name='noname', PY=-1, PY_std=-1, apr=-1, msat=-1):
        # Set the initial important data.
        self.name = name
        # sequence is the non-template strand == the "same as RNA-strand"
        self.sequence = sequence + 'TAAATATGGC'
        self.PY = PY
        self.PY_std = PY_std
        self.APR = apr
        self.msat = int(msat)

        # For N25 from Vo et. al 2003
        # 63% unproductive is about 1.75 times more than unproductive + productive
        self.unproductive_pct_yield = np.asarray([63.0, 16.0, 9.0, 2.2, 3.4,
            1.5, 3.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # These are optional data which are used if the raw quantitations are available
        self.quantitations = []
        # all data lists go from RNA-mer 2 to 21
        self.abortiveProb = []  # mean
        self.abortiveProb_std = []  # std

        # Raw quants
        self.rawAbortive = {}       # one entry for each quantitation
        self.rawAbortiveMean = -1   # one entry for each quantitation
        self.rawAbortiveStd = -1    # one entry for each quantitation
        self.abortive_pct = -1  # percent abortive transcript

        # FL
        self.fullLength = {}
        self.fullLengthMean = -1
        self.fullLengthStd = -1
        self.full_length_pct = -1  # percent FL transcript

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

    def averageRawDataAndFL(self):
        """
        More averaging and stding
        """

        if not self.sane():
            return

        # average mean
        self.fullLengthMean = np.mean(self.fullLength.values())
        self.fullLengthStd = np.std(self.fullLength.values())

        # average raw data
        self.rawAbortiveMean = np.mean(self.rawAbortive.values(), axis=0)
        self.rawAbortiveStd = np.std(self.rawAbortive.values(), axis=0)

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
        Use the raw data to calculate PY (mean and std)
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

        # self.abortive_pct is for all RNA; you just want the abortive %
        pct_abortive = self.rawAbortiveMean * 100 / sum(self.rawAbortiveMean)

        abortive_distribution_after_2nt = self.rawAbortiveMean[1:] / sum(self.rawAbortiveMean[1:])

        # estimate non-productive abortive % for 2nt RNA (dinucleoside tetraphosphate?)
        nonprod_2nt_est = pct_abortive[0] * 1.75
        nonprod_2nt_est = min(nonprod_2nt_est, 80.0)  # None above 80% (Vo 2003)
        nonprod_2nt_est = max(nonprod_2nt_est, 60.0)  # None below 60% (Vo 2003)

        # what remains to be spread over the rest
        remaining = 100. - nonprod_2nt_est
        the_rest = remaining * abortive_distribution_after_2nt

        # nonproductive pct
        nonprod_pct = np.asarray([nonprod_2nt_est] + the_rest.tolist())
        # So, 0.09 % of abortive RNA comes from the +18 position

        ap = calc_abortive_probability(nonprod_pct)

        assert sum(ap > 1) == 0

        self.unproductive_ap = ap

    def calc_AP(self):
        """
        Use the raw data to calculate AP (mean and std)
        """

        if not self.sane():
            return

        # store the AP for each quantitation
        self._APraw = {}
        self.totAbort = {}
        self.totRNA = {}

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

        self.abortiveProb = np.nanmean(self._APraw.values(), axis=0)
        self.abortiveProb_std = np.nanstd(self._APraw.values(), axis=0)

        self.totAbortMean = np.nanmean(self.totAbort.values(), axis=0)

        # test, no abortive probability should be less than zero
        assert sum([1 for ap in self.abortiveProb if ap < 0]) == 0

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

    def calc_AbortiveYield(self):
        """
        Calculate abortive to productive ratio

        """
        if not self.sane():
            return

        # store the PY for each quantitation
        self._AYraw = {}

        for quant in self.quantitations:

            totalRNA = sum(self.rawAbortive[quant]) + self.fullLength[quant]
            self._AYraw[quant] = self.rawAbortive[quant]/totalRNA

        self.AY = np.mean([py for py in self._AYraw.values()])
        self.AY_std = np.std([py for py in self._AYraw.values()])

    def calc_pct_yield(self):

        # You have to have initialized these values
        assert self.fullLengthMean != -1

        total_rna = sum(self.rawAbortiveMean) + self.fullLengthMean

        self.full_length_pct = 100 * self.fullLengthMean / total_rna
        self.abortive_pct = 100 * self.rawAbortiveMean / total_rna


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
    normdata = data / sum(data)

    # Find frac RNAP remaining up until position i
    frac_RNAP = np.asarray([np.sum(normdata[i:]) for i in range(data.size)])

    # Round. so that 0.001 becomes 0; so that 0.99999 becomes 1
    frac_RNAP = np.round(frac_RNAP, 6)

    # Calc the APs
    ap = normdata / frac_RNAP

    # nan likely means 0/0
    ap[np.isnan(ap)] = 0

    # We don't need no digits
    ap = np.round(ap, 2)

    return ap


if __name__ == '__main__':

    test_its = ITS('GATTACAGATTACAGATTACA', name='test_sequence')
    test_its.forward_transclocation(method='hein_et_al')

