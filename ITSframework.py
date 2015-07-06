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

        # These are optional data which are used if the raw quantitations are available
        self.quantitations = []
        # all data lists go from RNA-mer 2 to 21
        self.abortiveProb = []  # mean
        self.abortiveProb_std = []  # std

        # Raw quants
        self.rawData = {}  # one entry for each quantitation
        self.rawDataMean = -1  # one entry for each quantitation
        self.rawDataStd = -1   # one entry for each quantitation

        # FL
        self.fullLength = {}
        self.fullLengthMean = -1
        self.fullLengthStd = -1

        # helper arrays
        self._prctYield = {}
        self._RNAprodFromHere = {}

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
        self.rawDataMean = np.mean(self.rawData.values(), axis=0)
        self.rawDataStd = np.std(self.rawData.values(), axis=0)

    def sane(self):
        """
        Verify that raw abortive data and full length data are in place
        """

        if self.rawData == []:
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

            totalRNA = sum(self.rawData[quant]) + self.fullLength[quant]
            self._PYraw[quant] = self.fullLength[quant]/totalRNA

        self.PY = np.mean([py for py in self._PYraw.values()])
        self.PY_std = np.std([py for py in self._PYraw.values()])

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

            # define tmp arrays to work with
            rawD = self.rawData[quant]

            # Set all missing values to nan. Use nanmean to get watcha want.
            rawD = np.array([np.nan if x == -99 else x for x in rawD])

            fullL = self.fullLength[quant]

            totalAbortive = np.nansum(rawD)
            totalRNA = totalAbortive + fullL

            # Save these two for posterity
            self.totAbort[quant] = totalAbortive
            self.totRNA[quant] = totalRNA

            rDRange = range(len(rawD))

            # percentage yield at this x-mer
            prctYield = [(rD/totalRNA)*100. for rD in rawD]

            # amount of RNA produced from here and out
            RNAprodFromHere = [np.nansum(rawD[i:]) + fullL for i in rDRange]

            # percent of RNAP reaching this position
            prctRNAP = [(rpfh/totalRNA)*100. for rpfh in RNAprodFromHere]

            # probability to abort at this position
            # More accurately: probability that an abortive RNA of length i is
            # produced (it may backtrack and be slow to collapse, producing
            # little abortive product but reducing productive yield)
            self._APraw[quant] = [prctYield[i]/prctRNAP[i] for i in rDRange]

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

            totalRNA = sum(self.rawData[quant]) + self.fullLength[quant]
            self._AYraw[quant] = self.rawData[quant]/totalRNA

        self.AY = np.mean([py for py in self._AYraw.values()])
        self.AY_std = np.std([py for py in self._AYraw.values()])


if __name__ == '__main__':

    test_its = ITS('GATTACAGATTACAGATTACA', name='test_sequence')
    test_its.forward_transclocation(method='hein_et_al')

