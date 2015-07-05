import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
import data_handler
from scipy.stats import spearmanr, pearsonr, nanmean, nanstd, nanmedian  # NOQA
import os

# Specify better colors than the 'ugh' default colors in matplotlib
import brewer2mpl
bmap = brewer2mpl.get_map('Set1', 'qualitative', 9)
brew_red, brew_blue, brew_green, brew_purple, brew_orange, brew_yellow, brew_brown, brew_pink, brew_gray = bmap.mpl_colors

# Global settings for matplotlib
from matplotlib import rcParams
#rcParams['axes.labelsize'] = 9
#rcParams['xtick.color'] = 'gray'
#rcParams['ytick.labelsize'] = 9
#rcParams['legend.fontsize'] = 9

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times']
rcParams['text.usetex'] = True


class APstruct(object):
    def __init__(self, keq, ap, dg3d=-1, dgDna=-1, dgRna=-1):
        self.keq = keq
        self.dg3d = dg3d
        self.ap = ap
        self.dgDna = dgDna
        self.dgRna = dgRna


def ap_distribution(dg100, dg400):
    """
    Distribution of AP across all ITSs
    """

    from pandas import DataFrame as df

    for dset_name, ITSs in [('DG100', dg100), ('DG400', dg400)]:
        #if dset_name == 'DG400':
            #for its in ITSs:
                #its.abortiveProb = its.abortiveProb*2
        dsetmean = np.nanmean([i.PY for i in ITSs])
        if dset_name == 'DG100':
            rna_range = range(2, 21)
        else:
            rna_range = range(2, 16)

        for partition in ['low PY', 'high PY', 'all']:
            my_df = df()
            for rna_len in rna_range:
                if partition == 'low PY':
                    my_df[rna_len] = [i.abortiveProb[rna_len-2] for i in ITSs
                                      if i.PY < dsetmean]
                if partition == 'high PY':
                    my_df[rna_len] = [i.abortiveProb[rna_len-2] for i in ITSs
                                      if i.PY > dsetmean]
                if partition == 'all':
                    my_df[rna_len] = [i.abortiveProb[rna_len-2] for i in ITSs]

            # replace no signal with Nan. This way, you average those abortive
            # probabilities which actually exist (at position 20, 90% of itss
            # do not abort, but take the average AP of those who actually
            # reach)
            my_df = my_df.replace(0.0, np.nan)
            ax = my_df.plot(kind='box', ylim=(0, 0.8))
            fig = ax.get_figure()
            fig.suptitle('AP distributions for {0} {1}'.format(dset_name, partition))
            filepath = os.path.join('AP_distributions', dset_name + '_' + partition + '.pdf')
            fig.savefig(filepath, format='pdf', size_inches=(9, 15))


def raw_data_distribution(dg100, dg400):
    """
    Distribution of raw radiactive intensity across all ITSs
    """
    from pandas import DataFrame as df

    for dset_name, ITSs in [('DG100', dg100), ('DG400', dg400)]:
        #if dset_name == 'DG400':
            #for its in ITSs:
                #its.abortiveProb = its.abortiveProb*2
        dsetmean = np.nanmean([i.PY for i in ITSs])
        if dset_name == 'DG100':
            rna_range = range(2,21)
            ymax = 2*10**7
        else:
            rna_range = range(2,16)
            ymax = 0.2*10**7
        for division in ['low PY', 'high PY', 'all']:
            my_df = df()
            for rna_len in rna_range:
                if division == 'low PY':
                    my_df[rna_len] = [i.rawDataMean[rna_len-2] for i in ITSs
                            if i.PY < dsetmean]
                if division == 'high PY':
                    my_df[rna_len] = [i.rawDataMean[rna_len-2] for i in ITSs
                            if i.PY > dsetmean]
                if division == 'all':
                    my_df[rna_len] = [i.rawDataMean[rna_len-2] for i in ITSs]

            # -99 is nan here.
            my_df.replace('-99.0', value=np.nan, inplace=True)
            ax = my_df.plot(kind='box', ylim=(0, ymax))
            #ax = my_df.plot(kind='box')
            fig = ax.get_figure()
            fig.suptitle('Intensity distributions for {0} {1}'.
                    format(dset_name, division))
            dirname = 'Intensity_distributions'
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            filepath = os.path.join(dirname, dset_name + '_' +division + '.pdf')
            fig.savefig(filepath, format='pdf', size_inches=(9,15))


def basic_info(ITSs):
    """
    It seems that sum(AP) early is almost positively correlated with PY! It's
    only when considering the late AP that the correlation between sum(AP) and
    PY becomes negative, like you'd expect.
    XXX
    High PY variants are likely to abort early, but unlikely to abort late?
    XXX

    The AP after the 10-mer is predictive of PY. AP before 10-mer is not very
    predictive. This is at first appears counter-intuitive, since most abortive
    product is short. However, this must be balanced by the fact that it is
    more detrimental to PY to abort late. Perhaps late abortive products take
    longer time to release? When a full RNA-DNA hybrid is formed, backtracking
    and bubble collapse could be slower.

    So it has both taken longer time to produce the product, but it also takes
    longer time to backtrack and abort.

    What underlies the difference? Different stresses causing late and early
    collapse? Or just low/high probability of late collapse.
    """

    py = [i.PY for i in ITSs]

    # XXX RESULT: the "A" part contributes more to correlation with PY
    # than the "G" part of the pyrimidines for dg100, and much more for dg400.
    # for DG400 correlation is significant starting at +4, dips at +5, and then
    # increases gradually to +11. Correlation with G isn't significant at all
    # until + 10 or so! But gradually increases from +8 all the way up to + 15.
    # for DG400 G is actually significantly ANTI correlated with PY at +3 and +4!
    # this is also true for DG100, but not significant, and -0.18 or so.
    # The DG400 stuff can be an artefact from biasing toward the 3rd and 4th
    # nucleotides by forcing the dinucleotide structure.
    # Still, i'd say that for DG100 your conclusion still stands. How big is
    # the AG difference in Hein and Malinen?
    # Considering the U > C > A > G for going to the backtracked state, for
    # DG100 U has a stronger anti-correlation with PY than
    # There is an assymetry here though:
    #   A is more strongly correlated than G -> together they are stronger
    #   C and T are similarly anti-correlated -> together they are stronger (or
    #   just the negative of the AG correlation)

    # This can be result 1 in your work. Strong correlation with number of
    # purines. Has been shown ... G..., butstronger with A than with G.
    # Further, no correlation between A and G; if A and G were equally
    # beneficial to avoid paused and backtracked state, one would expect these
    # two values to have some correlation since they would be equally likely to
    # appear in the high-PY variants, and equally likely not to appear in the
    # low-PY variants.

    #ladder_pre = [[1 if nuc in ['G', 'A'] else 0 for nuc in i.sequence[:i.msat]] for i in ITSs]
    ladder_pre = [[1 if nuc in ['A'] else 0 for nuc in i.sequence[:20]] for i in ITSs]
    #ladder = [sum(ladder_pre[:i])]
    #xx=15
    #ladder_G = [[1 if nuc in ['G'] else 0 for nuc in i.sequence[:20]] for i in ITSs]
    #ladder_A = [[1 if nuc in ['A'] else 0 for nuc in i.sequence[:20]] for i in ITSs]
    #ladder_C = [[1 if nuc in ['C'] else 0 for nuc in i.sequence[:20]] for i in ITSs]
    #ladder_T = [[1 if nuc in ['T'] else 0 for nuc in i.sequence[:20]] for i in ITSs]
    #ladder_G = [[1 if nuc in ['G'] else 0 for nuc in i.sequence[:i.msat]] for i in ITSs]
    #ladder_A = [[1 if nuc in ['A'] else 0 for nuc in i.sequence[:i.msat]] for i in ITSs]
    #ladder_C = [[1 if nuc in ['C'] else 0 for nuc in i.sequence[:i.msat]] for i in ITSs]
    #ladder_T = [[1 if nuc in ['T'] else 0 for nuc in i.sequence[:i.msat]] for i in ITSs]
    #lad_np_G = [sum(ladder_G[i][:xx]) for i in range(len(ladder_pre))]
    #lad_np_A = [sum(ladder_A[i][:xx]) for i in range(len(ladder_pre))]
    #lad_np_C = [sum(ladder_C[i][:xx]) for i in range(len(ladder_pre))]
    #lad_np_T = [sum(ladder_T[i][:xx]) for i in range(len(ladder_pre))]
    #msats = [i.msat for i in ITSs]

    #nts = ('G', 'A', 'C', 'T')
    #nr_nts = (lad_np_G, lad_np_A, lad_np_C, lad_np_T)
    #for nt, nr_nt in zip(nts, nr_nts):
        #print nt, spearmanr(msats, nr_nt)

    #debug()
    avg_corr = []
    np_corr = []

    coeff = [0.25, 0.0, 0.55]
    for i in ITSs:
        i.calc_keq(*coeff, msat_normalization=True, rna_len=20)

    #rna_lenghts = range(2,10)

    #for x in rna_lenghts:
        #lad_np = [sum(ladder_pre[i][:x]) for i in range(len(ladder_pre))]
        ##avg_kbt = [np.mean(i.keq[:x]) for i in ITSs]
        #measure_values = [np.nanmean(i.keq[:x-1]) for i in ITSs]
        #np_corr.append(spearmanr(py, lad_np)[0])
        #avg_corr.append(spearmanr(py, measure_values)[0])

    rna_lenghts = range(12,16)
    for x in rna_lenghts:
        lad_np = [sum(ladder_pre[i][12:x]) for i in range(len(ladder_pre))]
        #avg_kbt = [np.mean(i.keq[:x]) for i in ITSs]
        measure_values = [np.nanmean(i.keq[12:x-1]) for i in ITSs]
        np_corr.append(spearmanr(py, lad_np)[0])
        avg_corr.append(spearmanr(py, measure_values)[0])

    fig, ax = plt.subplots()
    ax.plot(rna_lenghts, [-i for i in avg_corr], label='Average $K_{bt}$ up to RNA length')
    ax.plot(rna_lenghts, np_corr, label='Number of purines up to RNA length')

    ax.set_xlabel('RNA length')
    ax.set_ylabel('Spearman correlation with PY')
    ax.grid()

    ax.legend(loc='best')
    plt.show()


def greB_analysis(dg100):
    """
    Analyse the GreB+/- data from 2006 paper.
    """
    # These values are pretty different from the PY from quantitative data. How are they
    # obtained?
    table2_py_data = {
            'N25/A1': {'plus':     16.3, 'minus': 7.1},
            'N25': {'plus':        18.5, 'minus': 9.0},
            'DG115a': {'plus':     15.2, 'minus': 5.1},
            'DG127': {'plus':      13.4, 'minus': 3.5},
            'DG133': {'plus':      15.1, 'minus': 4.3},
            'N25anti': {'plus':    20.1, 'minus': 3.8},
            'DG137a': {'plus':     5.4,  'minus': 1.5},
            'DG154a': {'plus':     6.0,  'minus': 1.8},
            'N25/A1anti': {'plus': 6.0,  'minus': 1.5}}

    plus = [16.3, 18.5, 15.2, 13.4, 15.1, 20.1, 5.4, 6.0, 6.0]
    minus = [7.1, 9.0, 5.1, 3.5, 4.3, 3.8, 1.5, 1.8, 1.5]

    # XXX For the values in the table, the coefficient of variation is 30%
    # higher for GreB minus (statistically significant?). This
    # indicates that when using GreB, PY values become more similar. This
    # enforces the idea that abortive cycling drives low PY.
    # NOTE: the PY values given in Table 2 are high compared to the values
    # given in Table 1 of the paper. Using the values in Table 1, the GreB+
    # experiments have half the spread in PY compared to Table2.
    # You can mention this to Lilian. This is an argument that rate of
    # backtracking is not greatly sequence dependent.

    mean_plus = np.mean(plus)
    std_plus = np.std(plus)
    mean_minus = np.mean(minus)
    std_minus = np.std(minus)
    print(std_plus / mean_plus)
    print(std_minus / mean_minus)
    #return
    py_table1 = [i.PY for i in dg100 if i.name in table2_py_data]
    mean_table1 = np.mean(py_table1)
    std_table1 = np.std(py_table1)
    print(std_table1 / mean_table1)

    # Print PY for the ones you have GreB PY for
    purines = []
    for its in dg100:
        its.calc_purines()
        purines.append(sum(its.purines))
    for its in dg100:
        if its.name in table2_py_data:
            tabPy = table2_py_data[its.name]['minus']
            tabPyplus = table2_py_data[its.name]['plus']
            print(its.name + ' ', its.PY * 100, tabPy, tabPyplus)
            print(its.sequence)


def dataset_to_file(dg100, name, write_AP=False):
    if write_AP:
        filepath = name + '_parameters_AP.txt'
    else:
        filepath = name + '_parameters.txt'

    filehandle = open(filepath, 'wb')
    for its in dg100:
        line = []
        line.append(its.name)
        line.append(its.sequence)
        line.append(str(its.PY))

        if write_AP:
            for ap in its.abortiveProb:
                line.append(str(ap))

        filehandle.write('\t'.join(line) + '\n')

    filehandle.close()


def consensus_pause_sites(dg100, dg400):
    """
    Consensus pause signal is -10G, -1Y, +1G. Assume it's because of the RNA-DNA
    hybrid, can you find this signal in the ITSs?

    'G[GATC]{8}[T,C]G' ?

    Result: vague preference for low to medium PY ITSs, but one example in DG100 of a
    high PY ITS. Conclusion: does not have explanatory power. A lot more
    sequences have 2 out of 3 matching, but neither here is there a clear
    pattern.
    """
    import re

    p = re.compile('G[GATC]{8}[T,C]G')
    #matches = p.findall('GGGGGGGGGCG')
    #nr_matches = len(matches)

    for name, dset in [('DG100', dg100), ('DG400', dg400)]:

        ma = []
        PY = []

        print('\n {0}\n'.format(name))

        for its in sortITS(dset):
            matches = p.findall(its.sequence[:15])
            nr_matches = len(matches)
            print its.PY, nr_matches

            ma.append(nr_matches)
            PY.append(its.PY)

        print(spearmanr(ma, PY))


def sortITS(ITSs, attribute='PY'):
    """
    Sort the ITSs
    """

    ITSs = sorted(ITSs, key=attrgetter(attribute))

    return ITSs


def purine_vs_PY_scatter(dg100, dg400):
    """
    Reviewers want to see purine vs PY scatterplot. Do it for purines up to
    +15.
    """

    rcParams['figure.figsize'] = 8, 4  # This was the only way I could get this respected
    fig, axes = plt.subplots(ncols=2)
    ax_index = 0
    for dset_name, dset in [('DG100', dg100), ('DG400', dg400)]:
        purines = []
        PYs = []
        PYstd = []
        for its in dset:
            its.calc_purines()
            purines_15 = sum(its.purines[:15])

            purines.append(purines_15)
            PYs.append(its.PY)
            PYstd.append(its.PY_std)

        ax = axes[ax_index]
        ax.errorbar(purines, PYs, yerr=PYstd, fmt=None, ecolor=brew_gray, zorder=1,
                elinewidth=0.3)
        ax.scatter(purines, PYs, c='black', s=14, zorder=2, lw=0)
        ax.set_xlabel('Number of purines from +1 to +15')
        ax.set_ylabel('PY')
        ax.set_title(dset_name + ' library')

        if ax_index == 0:
            label = 'A'
        elif ax_index == 1:
            label = 'B'

        xpos = 0.03
        ypos = 0.97
        ax.text(xpos, ypos,
                label, transform=ax.transAxes, fontsize=18,
                fontweight='bold', va='top')

        ax_index += 1

    plt.tight_layout()
    #fig.savefig('figures/purine_py_scatter.pdf', format='pdf')
    fig.savefig('../../../The-Tome/my_papers/rna-dna-paper/supplementary/figures/purine_py_scatter.pdf', format='pdf')


def fold_difference(ITSs):
    """
    Establish that there is much more variation in abortive RNA than in full
    length RNA, showing that the difference in PY between different promoteres
    is driven primarily by a difference in abortive yield.
    """

    PY = [i.PY for i in ITSs]
    FL = [i.fullLengthMean for i in ITSs]
    AR = [i.totAbortMean for i in ITSs]
    SE = [i.SE for i in ITSs]

    print('Correlation PY and FL:')
    print spearmanr(PY, FL)

    print('Correlation PY and Abortive:')
    print spearmanr(PY, AR)
    print('')

    print('Correlation SE and FL:')
    print spearmanr(SE, FL)

    print('Correlation SE and Abortive:')
    print spearmanr(SE, AR)
    print('')

    max_FL = np.mean([i.fullLengthMean for i in ITSs[-3:]])
    min_FL = np.mean([i.fullLengthMean for i in ITSs[:3]])
    print('Full Length fold: {0}'.format(min_FL/max_FL))

    max_AR = np.mean([i.totAbortMean for i in ITSs[-3:]])
    min_AR = np.mean([i.totAbortMean for i in ITSs[:3]])
    print('Abortive RNA fold: {0}'.format(min_AR/max_AR))


def data_overview(ITSs):
    """
    This is the deleted stuff

    Plots the FL, PY, total Abortive and total RNA for all ITSs
    """

    #hooksMean = [('PY', '_PYraw'), ('FL', 'fullLength'),
            #('Total Abortive', 'totAbort'), ('Total RNA', 'totRNA')]
    hooksMean = [('PY', '_PYraw'), ('FL', 'fullLength'),
            ('Total Abortive', 'totAbort')]

    # separate the different quantitations
    fig, axes = plt.subplots(len(hooksMean))

    names = [i.name for i in ITSs]
    xvals = range(1, len(names) + 1)

    dsets = set(sum([i.quantitations for i in ITSs], []))
    # strip the N25 controls
    quantitations = [d for d in dsets if not d.endswith('.1')]

    for pltNr, (plotTag, attribute) in enumerate(hooksMean):

        ax = axes[pltNr]

        vals = {}

        # 1 Plot the attribute value for each dataset
        #for quant in quantitations:

            #y = [getattr(i, attribute)[quant] for i in ITSs]
            #ax.plot(xvals, y, label=quant, linewidth=2, marker='o')

        # 1 Plot mean attribute value
        vals = [[getattr(i, attribute)[quant] for i in ITSs] for quant in
                quantitations]

        ymean = np.mean(vals, axis=0)
        ystd = np.std(vals, axis=0)

        ax.errorbar(xvals, ymean, label='Mean', linewidth=2, marker='o',
                yerr=ystd)

        ax.set_xticklabels(names, rotation=30)
        ax.set_xticks(xvals)

        ax.legend(loc='best')
        ax.set_title(plotTag)

    plt.tight_layout()
    plt.show()


def shifted_ap_keq(ITSs, start=2, upto=15, plusmin=5):
    """
    Sort values with the DNA DNA value up to that point

    Optionally subtract the RNA-DNA value (length 10 RNA-DNA)

    How to do it? Make a basic class for AP! This makes it super-easy to change
    sortings: just loop through a keyword.

    Return arrays of AP and DG3N values in different batches. For example, the
    values could be split in 2 or 3. What determines the difference in
    correlation between those two groups will then be how you split the

    Maybe ... just maybe you have to do this correlation on a
    position-by-position basis and use a weighting factor to control for.
    Another method. After. Then you cover both ranges.
    """

    # add objects here and sort by score
    objects = {}

    for pm in range(-plusmin, plusmin+1):
        apobjs = []
        for its in ITSs:
            for ap_pos in range(start, upto):

                keq_pos = ap_pos + pm
                # only use keq_pos for the same range as ap_pos
                if keq_pos in range(start, upto):

                    apobjs.append(APstruct(its.keq[keq_pos], its.abortiveProb[ap_pos]))

        objects[pm] = apobjs

    return objects


def plus_minus_keq_ap(dg100, dg400):
    """
    Make a histogram plot of keq correlations with AP +/- nts of Keq.
    """

    #Indicate in some way the statistically significant results.
    #Do one from 1: and one from 3:
    ITSs = dg100 + dg400

    # Try both versions: one where you don't consider results when the +/-
    # thing is above/below 3/14
    upto = 14
    start = 2
    plusmin = 10

    objects = shifted_ap_keq(ITSs, start=start, upto=upto, plusmin=plusmin)

    # make the plot
    fig, ax = plt.subplots()

    bar_heights = []
    p_vals = []
    x_tick_names = range(-plusmin, plusmin+1)

    for pm in x_tick_names:
        apObjs = objects[pm]
        aps, keqs = zip(*[(apObj.ap, apObj.keq) for apObj in apObjs])
        corr, pval = spearmanr(aps, keqs)

        bar_heights.append(corr)
        p_vals.append(pval)

    xticks = range(1, len(x_tick_names)+1)
    plim = 0.05/len(xticks)  # simple bonferroni testing
    colors = []
    for pval in p_vals:
        if pval < plim:
            colors.append('g')
        else:
            colors.append('k')

    ax.bar(left=xticks, height=bar_heights, align='center', color=colors)
    ax.set_xticks(xticks)

    namesminus = [str(x) for x in range(-plusmin, 1)]
    namesplus = ['+'+str(x) for x in range(1, plusmin+1)]

    ax.set_xticklabels(namesminus + namesplus)

    ax.set_ylabel('Spearman correlation coefficient between AP and Keq')
    ax.set_xlabel('Position of Keq value relative to AP value')
    ax.set_title('Correlation between Keq and Abortive Probability '
                 'at shifted positions')
    ax.yaxis.grid()


def positionwise_shifted_ap_keq(ITSs, x_merStart, x_merStop, plusmin, keqAs='keq'):
    """
    Return array from 'x-mer' to 'y-mer'; each array has all the
    AP at those positions AND the Keq shifted by plusmin from those positions.

    Normally use keq for correlation, but also accept dg3d
    """

    # add objects here and sort by score
    objects = {}

    # in the its-arrays, abortiveProb[0] is for 2-mer
    # similarly, keq[0] is for the 2-nt RNA
    # therefore, if x_merStart is 2; we need to access ap_pos 0
    # in genral, it would have been nice if its's were indexed with x-mer
    arrayStart = x_merStart-2
    arrayStop = x_merStop-2
    for ap_pos in range(arrayStart, arrayStop):

        ap_val = []
        keqshift_val = []

        keq_pos = ap_pos + plusmin
        if keq_pos in range(arrayStart, arrayStop):
            for its in ITSs:

                ap_val.append(its.abortiveProb[ap_pos])
                if keqAs == 'keq':
                    keqshift_val.append(its.keq[keq_pos])
                elif keqAs == 'dg3d':
                    keqshift_val.append(its.dg3d[keq_pos])

        objects[ap_pos+2] = {'ap': ap_val, 'keq': keqshift_val}

    return objects


def position_wise_correlation_shifted(dg100, dg400):
    """
    Obtain the correlation between Keq and AP at each position from start to
    upto. At each position also shift the relationship: find out if the AP at a
    given position for all ITSs is correlated with a keq at another position.

    plan: the same as you get now, but obtain lists for each position 2-mer and
    3-mer and up, but produce 6 plots from -5 to +5 shift, and remember to
    bonferroni correct for each its-position. probably you should correct for
    each position * 6.

    When doing so, you obtain a strong, consistent negative correlation between
    the AP at pos and the Keq at pos+1.

    Looking at the scatter-plot, we see that when Keq is high at pos+1, then AP
    is invariably low at pos. When Keq is low, AP can be both low and high.

    So: pretranslocated state is favored at pos+1, means AP is low at pos.

    How do we explain this?

    Normalizing the AP changes the correlation between sum(SE) and sum(AP).
    Before normalization, the correlation is positive, indicating probably that
    sum(AP) is strongly correlated with PY. After normalization sum(AP) is
    NEGATIVELY correlated with sum(SE). The normalization is making the AP
    values more position-rich in information.

    It is not strange that the sign of the correlation between the positions
    and Keq do not change with normalization.

    What is strange is that there is a negative correlation at all.
    """

    #ITSs = dg400 + dg100
    ITSs = dg400
    #ITSs = dg100

    x_merStart = 2
    x_merStop = 15

    plusmin = 1
    pm_range = range(-plusmin, plusmin+1)

    for (dg, ITSs) in [('DG100', dg100), ('DG400', dg400), ('Both', dg100+dg400)]:

        # make one figure per pm val
        for pm in pm_range:
            objects = positionwise_shifted_ap_keq(ITSs, x_merStart=x_merStart,
                    x_merStop=x_merStop, plusmin=pm, keqAs='dg3d')

            fig, ax = plt.subplots()

            bar_heights = []
            p_vals = []

            nr_corl = 0

            for xmer, vals in sorted(objects.items()):

                keqs = vals['keq']
                aps = vals['ap']

                # if the shift of keq is too far away from ap, no values will be
                # returned
                if keqs != []:

                    corr, pval = spearmanr(keqs, aps)

                    bar_heights.append(corr)
                    p_vals.append(pval)

                    nr_corl += 1  # nr of non-zero correlations for bonferroni
                else:
                    bar_heights.append(0)
                    p_vals.append(0)

                # see the scatter plots for the +1 case
                #if pm == 1:
                    #figu, axu = plt.subplots()
                    #axu.scatter(keqs, aps)
                    #axu.set_xlabel('Keq at pos {0}'.format(xmer+pm))
                    #axu.set_ylabel('AP at pos {0}'.format(xmer))

                #if xmer == 2 and pm == 0:

            # the positions along the x-axis (got nothing to do with labels)
            xpos = range(1, len(bar_heights)+1)
            # this is affected by xlim

            #plim = 0.05/(nr_corl*len(pm_range))  # simple bonferroni testing
            plim = 0.05/(nr_corl)  # simple bonferroni testing

            colors = []
            for pval in p_vals:
                if pval < plim:
                    colors.append('g')
                else:
                    colors.append('k')

            ax.bar(left=xpos, height=bar_heights, align='center', color=colors)
            ax.set_xticks(xpos)  # to make the 'ticks' physically appear
            ax.set_xlim(0, x_merStop-2)

            xticknames = range(x_merStart, x_merStop+1)
            ax.set_xticklabels(xticknames)

            ax.set_ylabel('Spearman correlation coefficient between AP and shifted Keq')
            ax.set_xlabel('The AP of corresponding x-mer')
            if pm < 0:
                shift = str(pm)
            else:
                shift = '+' + str(pm)

            ax.set_title('{0}: Keq shifted by {1}'.format(dg, shift))
            ax.yaxis.grid()

            ax.set_ylim(-0.81, 0.81)
            ax.set_yticks(np.arange(-0.8, 0.9, 0.2))


def apRawSum(dg100, dg400):
    """
    Simple bar plot to show where the raw reads are highest and where the
    abortive probabilities are highest across all variants.

    Result: 6 and 8 are big values. Wonder if Lilian has seen smth like this.

    Big steps at 6 and 13 in AP. These sites correspond to big jumps also in
    your ladder-plot.

    I think Lilian should see these plots. I don't think she has seen them
    before. Make them when you send her an email.
    """

    #ITSs = dg100
    ITSs = dg400
    #ITSs = dg400 + dg100

    mers = range(0, 19)
    x_names = range(2, 21)

    #rawSum = [sum([i.rawDataMean[x] for i in ITSs]) for x in mers]
    #apSum = [sum([i.abortiveProb[x] for i in ITSs]) for x in mers]

    rawMean = [np.mean([i.rawDataMean[x] for i in ITSs]) for x in mers]
    apMean = [np.mean([i.abortiveProb[x] for i in ITSs]) for x in mers]

    rawStd = [np.std([i.rawDataMean[x] for i in ITSs]) for x in mers]
    apStd = [np.std([i.abortiveProb[x] for i in ITSs]) for x in mers]

    for heights, stds, descr in [(rawMean, rawStd, 'Raw'), (apMean, apStd, 'AP')]:

        fig, ax = plt.subplots()

        ax.bar(left=mers, height=heights, align='center', yerr=stds)
        ax.set_xticks(mers)
        ax.set_xticklabels(x_names)

        ax.set_xlabel('X-mer')
        ax.set_ylabel(descr)


def abortive_bar(dg100, dg400):
    """
    Simply a bar-plot showing the amount of abortive product
    """

    fig, axes = plt.subplots(2)

    xmers = range(2, 21)
    attr = 'rawDataMean'

    for ax_nr, (title, dset) in enumerate([('DG100', dg100), ('DG400', dg400)]):

        ax = axes[ax_nr]

        bar_height = []

        for nc in xmers:
            bar_height.append(sum([getattr(i, attr)[nc-2] for i in dset]))

        ax.bar(left=xmers, height=bar_height, align='center')
        ax.set_xticks(xmers)
        ax.set_xlim(xmers[0]-1, xmers[-1])

        ax.set_xlabel('Nucleotide')
        ax.set_ylabel('Abortive product')
        ax.set_title(title)

    fig.suptitle('Sum of abortive product at each nucleotide position')


def get_movAv_array(dset, center, movSize, attr, prePost):
    """
    Hey, it's not an average yet: you're not dividing by movSize .. but you
    could.
    """

    movAr = []

    # the last coordinate at which you find keq and AP info
    for i in dset:

        # for exampe abortiveProb, Keq, etc.
        array = getattr(i, attr)

        idx = center-2
        # check if there is enough space around the center to proceed
        # if center is 1, and movSize is 3, nothing can be done, since you must
        # average center-3:center+4
        if (idx - movSize) < 0 or (idx + movSize + 1) > 18:
            movAr.append(None)  # test for negative values when plotting

        else:
            if prePost == 'both':
                movAr.append(sum(array[idx-movSize:idx+movSize+1]))
            elif prePost == 'pre':
                #movAr.append(sum(array[idx-movSize:idx+1]))
                movAr.append(sum(array[idx-movSize:idx]))
            elif prePost == 'post':
                #movAr.append(sum(array[idx+2:idx+movSize+1]))
                movAr.append(sum(array[idx+1:idx+movSize+1]))
            else:
                print ":(((())))"

    return movAr


def moving_average_ap(dg100, dg400):
    """
    Define a moving average size movSize and correlate each moving average
    window of either abortive probability or raw abortive product with the PY/FL/TA

    The results are insteresting: dg100 and dg400 have slightly different
    profiles
    """

    movSize = 0
    xmers = range(2, 21)
    #attr = 'abortiveProb'
    attr = 'rawDataMean'
    plim = 0.05

    #for title, dset in [('DG100', dg100), ('DG400', dg400),
            #('DG100 + DG400', dg100+dg400)]:
    for title, dset in [('DG100', dg100), ('DG400', dg400)]:

        py = [i.PY for i in dset]
        #fl = [i.fullLengthMean for i in dset]
        #ta = [i.totAbortMean for i in dset]

        # get a moving average window for each center_position
        # calculating from an x-mer point of view
        # Use a less conservative test for the paper

        #for label, meas in [('PY', py), ('FL', fl), ('TotalAbort', ta)]:
        for label, meas in [('PY', py)]:
        #for label, meas in [('FL', fl), ('TotalAbort', ta)]:

            # the bar height will be the correlation with the above three values for
            # each moving average center position
            bar_height = []
            pvals = []

            nr_tests = 0

            for mer_center_pos in xmers:
                movArr = get_movAv_array(dset, center=mer_center_pos,
                        movSize=movSize, attr=attr, prePost='both')

                corr, pval = spearmanr(meas, movArr)

                if not np.isnan(corr):
                    nr_tests += 1

                bar_height.append(corr)
                pvals.append(pval)

            # if no tests passed, make to 1 to avoid division by zero
            if nr_tests == 0:
                nr_tests = 1

            colors = []
            for pval in pvals:
                if pval < (plim/nr_tests):
                    colors.append('g')
                else:
                    colors.append('k')

            fig, ax = plt.subplots()
            ax.bar(left=xmers, height=bar_height, align='center', color=colors)
            ax.set_xticks(xmers)
            ax.set_xlim(xmers[0]-1, xmers[-1])

            ax.set_xlabel('Nucleotide')
            ax.set_ylabel('Correlation coefficient')
            #ax.set_title('{0} -- {1}: window size: '
                            #'{2}'.format(label, attr, movSize))
            ax.set_title('{0}: Correlations between sum of abortive product at each ITS'
                    ' position and PY'.format(title))


def various_analyses(dg100, dg400):
    """
    Misc. data analysis
    """

    # basic correlations
    basic_info(dg100)
    basic_info(dg400)

    #ap_distribution(dg100, dg400)
    #raw_data_distribution(dg100, dg400)
    #greB_analysis(dg100)

    #dataset_to_file(dg100, name='dg100', write_AP=True)
    #dataset_to_file(dg400, name='dg400')
    #return

    #consensus_pause_sites(dg100, dg400)

    # XXX: start here: create this figure, add to supplementary, build
    # supplementary, build new version, make a list of all your changes, and
    # send to Lilian et. al
    #purine_vs_PY_scatter(dg100, dg400)

    ## plot data when sorting by SE
    #ITSs = sortITS(ITSs, 'SE')

    #fold_difference(ITSs)
    ## Plot FL, PY, total RNA for the different quantitations
    #data_overview(ITSs)

    # ap keq for all positions
    #plus_minus_keq_ap(dg100, dg400)

    # ap keq at each position
    #position_wise_correlation_shifted(dg100, dg400)

    # bar plot of the sum of AP and the sum of raw
    #apRawSum(dg100, dg400)

    # abortive bar plot
    #plt.ion()
    #abortive_bar(dg100, dg400)

    # moving average of AP vs PY/FL/TA
    #moving_average_ap(dg100, dg400)


def main():

    dg100 = data_handler.ReadData('dg100-new')
    dg400 = data_handler.ReadData('dg400')

    various_analyses(dg100, dg400)


if __name__ == '__main__':
    main()
