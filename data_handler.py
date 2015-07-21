from __future__ import division
import pandas
from numpy import ndarray
import os

from ITSframework import ITS
from ipdb import set_trace as debug  # NOQA

# We store files relative to this directory
package_directory = os.path.dirname(os.path.abspath(__file__))


def StringOrFloat(incoming):
    """ Return float if element is float, otherwise return unmodified. """
    datatype = type(incoming)
    if datatype == str:
        try:
            fl00t = float(incoming)
            return fl00t
        except:
            return incoming
    elif (datatype is list) or (datatype is ndarray):
        outgoing = []
        for element in incoming:
            try:
                element = float(element)
                outgoing.append(element)
            except:
                outgoing.append(element)
        return outgoing

    raise ValueError('Input must be string/float/int or list/array of these. ')


def PYHsu_oldcsv(filepath):
    import csv
    """ Read Hsu dg100 csv-file with PY etc data. """
    f = open(filepath, 'rb')
    a = csv.reader(f, delimiter='\t')
    b = [[StringOrFloat(v) for v in row] for row in a]
    f.close()

    lizt = [[row[0], row[1], row[2], row[3], row[4], row[6], row[8], row[10],
             row[11]] for row in b]

    # Making a list of instances of the ITS class.
    ITSs = [ITS(row[1], row[0], row[2], row[3], row[6], row[7]) for row in lizt]

    return ITSs


def read_raw(path, files, dset, skipN25=False):
    """
    Read raw quantitation data into ITS objects and calculate relevant values.
    """

    # 1) test: assert that the names of the promoter variants are the same in
    # all the raw-files (and that the set size is identical)

    # 2) test assert that there is a match between the names in the sequence
    # file and the names in the should I?

    # XXX TODO HERE fix the N25/A1 anti stuff once and for all. Don't have
    # path deliminters in IDs.

    ITSs = {}
    seqs = {}

    # specify the location of the ITS DNA sequence
    if dset == 'dg400':
        seq_file = os.path.join(package_directory, 'sequence_data/tested_seqs_dg400.txt')

    if dset == 'dg100':
        seq_file = os.path.join(package_directory, 'sequence_data/Hsu/dg100Seqs')

    for line in open(seq_file, 'rb'):
        variant, seq = line.split()

        # special case to catch spelling
        if variant.endswith('A1anti'):
            seqs['N25-A1anti'] = seq
        else:
            seqs[variant] = seq

    for fNr, fName in files.items():
        target = os.path.join(path, fName)
        from_file = pandas.read_csv(target, index_col=0)

        # 2-mer to 22-mer, depending on the gel.
        labels = [i for i in from_file.index if not i.startswith('F')]

        for variant in from_file:

            # skip all N25s if requested
            if skipN25 and 'N25' in variant:
                continue

            # turn nans to zeros
            # XXX: this is an issue that remains: when using -99, you get into
            # situations like this: 13, 0, 0; average = 13, while it should be
            # less, because in 2 replicas there was no signal.
            # But, when you look at average AP for ALL its, you WANT to perform a nanmean
            # The solution is then to subsitute 0 for nan when doing that.
            entry = from_file[variant].fillna(value=-0)

            # deal with replicas that contain a '.'
            replica = ''
            if '.' in variant:
                variant, replica = variant.split('.')

            # Some N25antis have different names
            if 'N25anti' in variant:
                variant = 'N25anti'

            # if exists (even as replica), append to it
            if variant in ITSs:
                itsObj = ITSs[variant]
            # if not, create new
            else:
                if variant[:-1] in ITSs:
                    var = variant[:-1]
                    itsObj = ITSs[var]
                else:
                    itsObj = ITS(seqs[variant], name=variant)
                    ITSs[variant] = itsObj

            saveas = str(fNr)
            if replica != '':
                saveas = saveas + '.' + replica

            # get the raw reads for each x-mer
            rawReads = [entry[l] for l in labels]
            itsObj.rawAbortive[saveas] = rawReads
            itsObj.fullLength[saveas] = entry['FL']  # first gel has FL2?
            itsObj.quantitations.append(saveas)

    # calculate AP and PY
    for name, itsObj in ITSs.items():
        itsObj.labels = labels
        itsObj.calc_AP()
        itsObj.calc_AP_old()
        itsObj.calc_PY()
        #itsObj.calc_AP_unproductive()  # might come in handy later
        itsObj.calc_MSAT()

    return ITSs


def add_AP(ITSs):
    """
    Parse the AP file for the DG100 library and get the AP in there
    """
    dg100ap = 'Hsu_original_data/AbortiveProbabilities/abortiveProbabilities_mean.csv'
    full_path = os.path.join(package_directory, dg100ap)
    APs = pandas.read_csv(full_path, index_col=0).fillna(value=-999)
    for its in ITSs:
        its.abortiveProb = APs[its.name].tolist()

    return ITSs


def ReadDG100Old(path):
    ITSs = PYHsu_oldcsv(path)  # Unmodified Hsu data
    ITSs = add_AP(ITSs)

    return ITSs


def ReadData(dataset):
    """ Read Hsu data.

    Possible input is dg100, dg100-new, dg400.

    dg100 is just the PY values calculated by Lilian from the paper.

    dg100-new uses the raw transcription data. This allows you to calculate PY
    and AP yourself.

    dg400 also uses raw transcription data.
    """

    # Selecting the dataset you want to use
    if dataset == 'dg100':
        path = os.path.join(package_directory, 'sequence_data/Hsu/csvHsu')
        ITSs = ReadDG100Old(path)

    elif dataset == 'dg100-new':
        path = os.path.join(package_directory, 'Hsu_original_data/2006_paper/2013_email')
        files = {'1122_first':  'quant1122.csv',
                 '1207_second': 'quant1207.csv',
                 '1214_third':  'quant1214.csv'}

        # this re-calculates PY and AP and all that from raw data
        ITSs = read_raw(path, files, dset='dg100')

        # read its-data the "old" way just to get msat and copy msat to ITSs
        # read the 'new' way
        oldcsvpath = os.path.join(package_directory, 'sequence_data/Hsu/csvHsu')
        ITSs_oldcsv = PYHsu_oldcsv(oldcsvpath)
        for its_name in ITSs:
            #  Rename here as well ...
            for its_old in ITSs_oldcsv:
                if its_name == its_old.name:
                    ITSs[its_name].msat = its_old.msat

    elif dataset == 'dg400':
        path = os.path.join(package_directory, 'prediction_experiment/raw_data')
        files = {'16_first':  'quant16_raw.csv',
                 '27_second': 'quant27_raw.csv',
                 '23_first':  'quant23_raw.csv'}

        ITSs = read_raw(path, files, dset='dg400')
        # set msat on dg400 library to 21 just to be sure (not measured)
        for its in ITSs.values():
            its.msat = 21

    else:
        print('Provide valid dataset input to ReadData!')
        return 0

    # Make into a list for backwards compatability and sort the ITSs
    from operator import attrgetter as atrb
    #ITSs = sorted([obj for obj in ITSs.values()], key=atrb('name'))
    ITSs = sorted([obj for obj in ITSs.values()], key=atrb('PY'))

    return ITSs
