import os
import sys,re
import pickle
from collections import Counter
from functools import reduce
import itertools


ALPHABET='ACGT'


def readDNAFasta(file):
	if os.path.exists(file) == False:
		print('Error: "' + file + '" DNA sequence does not exist.')
		sys.exit(1)

	with open(file) as f:
		records = f.read()

	if re.search('>', records) == None:
		print('The input DNA sequence must be fasta format.')
		sys.exit(1)

	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ACGT-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta

def frequency(t1_str, t2_str):

    i, j, tar_count = 0, 0, 0
    len_tol_str = len(t1_str)
    len_tar_str = len(t2_str)
    while i < len_tol_str and j < len_tar_str:
        if t1_str[i] == t2_str[j]:
            i += 1
            j += 1
            if j >= len_tar_str:
                tar_count += 1
                i = i - j + 1
                j = 0
        else:
            i = i - j + 1
            j = 0

    return tar_count
def make_kmer_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k should >=0 and an interger type,alphabet should be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k should >=0")
        raise ValueError
def extend_phyche_index(original_index, extend_index):
    if extend_index is None or len(extend_index) == 0:
        return original_index
    for key in list(original_index.keys()):
        original_index[key].extend(extend_index[key])
    return original_index

def get_phyche_index(k, phyche_list):
    phyche_value = {}
    if 0 == len(phyche_list):
        for nucleotide in make_kmer_list(k, ALPHABET):
            phyche_value[nucleotide] = []
        return phyche_value

    nucleotide_phyche_value = get_phyche_factor_dic(k)
    for nucleotide in make_kmer_list(k, ALPHABET):
        if nucleotide not in phyche_value:
            phyche_value[nucleotide] = []
        for e in nucleotide_phyche_value[nucleotide]:
            if e[0] in phyche_list:
                phyche_value[nucleotide].append(e[1])

    return phyche_value
def get_phyche_value_psednc(extra_phyche_index=None):

    if extra_phyche_index is None:
        extra_phyche_index = {}

    original_phyche_value = {'AA': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11],
                             'AC': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
                             'AG': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
                             'AT': [1.07, 0.22, 0.62, -1.02, 2.51, 1.17],
                             'CA': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
                             'CC': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
                             'CG': [-1.66, -1.22, -0.44, -0.82, -0.29, -1.39],
                             'CT': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
                             'GA': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
                             'GC': [-0.08, 0.22, 1.33, -0.35, 0.65, 1.59],
                             'GG': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
                             'GT': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
                             'TA': [-1.23, -2.37, -0.44, -2.24, -1.51, -1.39],
                             'TC': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
                             'TG': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
                             'TT': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11]}

    phyche_value = extend_phyche_index(original_phyche_value, extra_phyche_index)

    return phyche_value
def get_phyche_value_pseknc(extra_phyche_index=None):
    """For PseDNC, PseKNC, make sequence_list and phyche_value.

    :param input_data: file type or handle.
    :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                     the value is its physicochemical property value (list).
                               It means the user-defined physicochemical indices.
    """
    if extra_phyche_index is None:
        extra_phyche_index = {}

    original_phyche_value = {
        'AA': [0.06, 0.5, 0.09, 1.59, 0.11, -0.11],
        'AC': [1.5, 0.5, 1.19, 0.13, 1.29, 1.04],
        'GT': [1.5, 0.5, 1.19, 0.13, 1.29, 1.04],
        'AG': [0.78, 0.36, -0.28, 0.68, -0.24, -0.62],
        'CC': [0.06, 1.08, -0.28, 0.56, -0.82, 0.24],
        'CA': [-1.38, -1.36, -1.01, -0.86, -0.62, -1.25],
        'CG': [-1.66, -1.22, -1.38, -0.82, -0.29, -1.39],
        'TT': [0.06, 0.5, 0.09, 1.59, 0.11, -0.11],
        'GG': [0.06, 1.08, -0.28, 0.56, -0.82, 0.24],
        'GC': [-0.08, 0.22, 2.3, -0.35, 0.65, 1.59],
        'AT': [1.07, 0.22, 0.83, -1.02, 2.51, 1.17],
        'GA': [-0.08, 0.5, 0.09, 0.13, -0.39, 0.71],
        'TG': [-1.38, -1.36, -1.01, -0.86, -0.62, -1.25],
        'TA': [-1.23, -2.37, -1.38, -2.24, -1.51, -1.39],
        'TC': [-0.08, 0.5, 0.09, 0.13, -0.39, 0.71],
        'CT': [0.78, 0.36, -0.28, 0.68, -0.24, -0.62]}

    phyche_value = extend_phyche_index(original_phyche_value, extra_phyche_index)
    return phyche_value
def generate_phyche_value(k, phyche_index=None, all_property=False, extra_phyche_index=None):
    
    if phyche_index is None:
        phyche_index = []
    if extra_phyche_index is None:
        extra_phyche_index = {}

    diphyche_list = ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content',
                     'A-philicity', 'Propeller twist', 'Duplex stability:(freeenergy)',
                     'Duplex tability(disruptenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                     'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
                     'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                     'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
                     'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
                     'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction', 'Twist', 'Tilt',
                     'Roll', 'Shift', 'Slide', 'Rise']
    triphyche_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
                      'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid', 'MW-Daltons',
                      'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']

    # Add and test physicochemical properties.
    if 2 == k:
        if all_property is True:
            phyche_index = diphyche_list
        else:
            for e in phyche_index:
                if e not in diphyche_list:
                    fault = 'The physicochemical property ' + e + ' cannot be found.'
                    import sys

                    sys.stderr.write(fault)
                    sys.exit(0)
    elif 3 == k:
        if all_property is True:
            phyche_index = triphyche_list
        else:
            for e in phyche_index:
                if e not in triphyche_list:
                    error_info = 'the physicochemical property ' + e + ' cannot be found.'
                    import sys

                    sys.stderr.write(error_info)
                    sys.exit(0)

    return extend_phyche_index(get_phyche_index(k, phyche_index), extra_phyche_index)
def get_phyche_factor_dic(k):
    full_path = os.path.realpath(__file__)
    if 2 == k:
        file_path = "%s/data/mmc3.data" % os.path.dirname(full_path)
    elif 3 == k:
        file_path = "%s/data/mmc4.data" % os.path.dirname(full_path)
    else:
        sys.stderr.write("The k can just be 2 or 3.")
        sys.exit(0)

    try:
        with open(file_path, 'rb') as f:
            phyche_factor_dic = pickle.load(f)
    except:
        with open(file_path, 'r') as f:
            phyche_factor_dic = pickle.load(f)

    return phyche_factor_dic

def parallel_cor_function(nucleotide1, nucleotide2, phyche_index):
    temp_sum = 0.0
    phyche_index_values = list(phyche_index.values())
    len_phyche_index = len(phyche_index_values[0])
    for u in range(len_phyche_index):
        temp_sum += pow(float(phyche_index[nucleotide1][u]) - float(phyche_index[nucleotide2][u]), 2)

    return temp_sum / len_phyche_index


def series_cor_function(nucleotide1, nucleotide2, big_lamada, phyche_value):

    return float(phyche_value[nucleotide1][big_lamada]) * float(phyche_value[nucleotide2][big_lamada])


def get_parallel_factor(k, lamada, sequence, phyche_value):

    theta = []
    l = len(sequence)

    for i in range(1, lamada + 1):
        temp_sum = 0.0
        for j in range(0, l - k - i + 1):
            nucleotide1 = sequence[j: j+k]
            nucleotide2 = sequence[j+i: j+i+k]
            temp_sum += parallel_cor_function(nucleotide1, nucleotide2, phyche_value)

        theta.append(temp_sum / (l - k - i + 1))

    return theta


def get_series_factor(k, lamada, sequence, phyche_value):

    theta = []
    l_seq = len(sequence)
    temp_values = list(phyche_value.values())
    max_big_lamada = len(temp_values[0])

    for small_lamada in range(1, lamada + 1):
        for big_lamada in range(max_big_lamada):
            temp_sum = 0.0
            for i in range(0, l_seq - k - small_lamada + 1):
                nucleotide1 = sequence[i: i+k]
                nucleotide2 = sequence[i+small_lamada: i+small_lamada+k]
                temp_sum += series_cor_function(nucleotide1, nucleotide2, big_lamada, phyche_value)

            theta.append(temp_sum / (l_seq - k - small_lamada + 1))

    return theta


def make_pseknc_vector(sequence_list, lamada, w, k, phyche_value, theta_type=1):

    kmer = make_kmer_list(k, ALPHABET)
    vector = []

    for sequence_ in sequence_list:
        name,sequence=sequence_[0],sequence_[1]
        if len(sequence) < k or lamada + k > len(sequence):
            error_info = "Sorry, the sequence length must be larger than " + str(lamada + k)
            sys.stderr.write(error_info)
            sys.exit(0)

        # Get the nucleotide frequency in the DNA sequence.
        fre_list = [frequency(sequence, str(key)) for key in kmer]
        fre_sum = float(sum(fre_list))

        # Get the normalized occurrence frequency of nucleotide in the DNA sequence.
        fre_list = [e / fre_sum for e in fre_list]

        # Get the theta_list according the Equation 5.
        if 1 == theta_type:
            theta_list = get_parallel_factor(k, lamada, sequence, phyche_value)
        elif 2 == theta_type:
            theta_list = get_series_factor(k, lamada, sequence, phyche_value)
        theta_sum = sum(theta_list)

        # Generate the vector according the Equation 9.
        denominator = 1 + w * theta_sum

        temp_vec = [round(f / denominator, 3) for f in fre_list]
        for theta in theta_list:
            temp_vec.append(round(w * theta / denominator, 4))

        vector.append(temp_vec)

    return vector


def get_parallel_factor_psednc(lamada, sequence, phyche_value):

    theta = []
    l = len(sequence)

    for i in range(1, lamada + 1):
        temp_sum = 0.0
        for j in range(0, l - 1 - lamada):
            nucleotide1 = sequence[j] + sequence[j + 1]
            nucleotide2 = sequence[j + i] + sequence[j + i + 1]
            temp_sum += parallel_cor_function(nucleotide1, nucleotide2, phyche_value)

        theta.append(temp_sum / (l - i - 1))

    return theta


def make_old_pseknc_vector(sequence_list, lamada, w, k, phyche_value, theta_type=1):

    kmer = make_kmer_list(k, ALPHABET)
    vector = []

    for sequence_ in sequence_list:
        name,sequence=sequence_[0],sequence_[1]
        if len(sequence) < k or lamada + k > len(sequence):
            error_info = "Sorry, the sequence length must be larger than " + str(lamada + k)
            sys.stderr.write(error_info)
            sys.exit(0)

        # Get the nucleotide frequency in the DNA sequence.
        fre_list = [frequency(sequence, str(key)) for key in kmer]
        fre_sum = float(sum(fre_list))

        # Get the normalized occurrence frequency of nucleotide in the DNA sequence.
        fre_list = [e / fre_sum for e in fre_list]

        # Get the theta_list according the Equation 5.
        if 1 == theta_type:
            theta_list = get_parallel_factor_psednc(lamada, sequence, phyche_value)
        elif 2 == theta_type:
            theta_list = get_series_factor(k, lamada, sequence, phyche_value)
        theta_sum = sum(theta_list)

        # Generate the vector according the Equation 9.
        denominator = 1 + w * theta_sum

        temp_vec = [round(f / denominator, 3) for f in fre_list]
        for theta in theta_list:
            temp_vec.append(round(w * theta / denominator, 4))

        vector.append(temp_vec)

    return vector


def make_ac_vector(sequence_list, lag, phyche_value, k):
    phyche_values = list(phyche_value.values())
    len_phyche_value = len(phyche_values[0])

    vec_ac = []
    for sequence_ in sequence_list:
        name,sequence=sequence_[0],sequence_[1]
        len_seq = len(sequence)
        each_vec = []

        for temp_lag in range(1, lag + 1):
            for j in range(len_phyche_value):

                # Calculate average phyche_value for a nucleotide.
                ave_phyche_value = 0.0
                for i in range(len_seq - temp_lag - k + 1):
                    nucleotide = sequence[i: i + k]
                    ave_phyche_value += float(phyche_value[nucleotide][j])
                ave_phyche_value /= len_seq

                # Calculate the vector.
                temp_sum = 0.0
                for i in range(len_seq - temp_lag - k + 1):
                    nucleotide1 = sequence[i: i + k]
                    nucleotide2 = sequence[i + temp_lag: i + temp_lag + k]
                    temp_sum += (float(phyche_value[nucleotide1][j]) - ave_phyche_value) * (
                        float(phyche_value[nucleotide2][j]))
                each_vec.append(round(temp_sum / (len_seq - temp_lag - k + 1), 3))
        vec_ac.append(each_vec)
    return vec_ac


def make_cc_vector(sequence_list, lag, phyche_value, k):
    phyche_values = list(phyche_value.values())
    len_phyche_value = len(phyche_values[0])
    vec_cc = []
    for sequence_ in sequence_list:
        name,sequence=sequence_[0],sequence_[1]
        len_seq = len(sequence)
        each_vec = []

        for temp_lag in range(1, lag + 1):
            for i1 in range(len_phyche_value):
                for i2 in range(len_phyche_value):
                    if i1 != i2:
                        # Calculate average phyche_value for a nucleotide.
                        ave_phyche_value1 = 0.0
                        ave_phyche_value2 = 0.0
                        for j in range(len_seq - temp_lag - k + 1):
                            nucleotide = sequence[j: j + k]
                            ave_phyche_value1 += float(phyche_value[nucleotide][i1])
                            ave_phyche_value2 += float(phyche_value[nucleotide][i2])
                        ave_phyche_value1 /= len_seq
                        ave_phyche_value2 /= len_seq
                        # Calculate the vector.
                        temp_sum = 0.0
                        for j in range(len_seq - temp_lag - k + 1):
                            nucleotide1 = sequence[j: j + k]
                            nucleotide2 = sequence[j + temp_lag: j + temp_lag + k]
                            temp_sum += (float(phyche_value[nucleotide1][i1]) - ave_phyche_value1) * \
                                        (float(phyche_value[nucleotide2][i2]) - ave_phyche_value2)
                        each_vec.append(round(temp_sum / (len_seq - temp_lag - k + 1), 3))

        vec_cc.append(each_vec)

    return vec_cc


def acc_property(input_data, k, phyche_index=None, all_property=False, extra_phyche_index=None):

    if phyche_index is None:
        phyche_index = []
    if extra_phyche_index is None:
        extra_phyche_index = {}
    property_value = generate_phyche_value(k, phyche_index, all_property, extra_phyche_index)

    return property_value
def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer

def RC(kmer):
    myDict = {
        'A': 'T',
        'C': 'G',
        'G': 'C',
        'T': 'A'
    }
    return ''.join([myDict[nc] for nc in kmer[::-1]])


def generateRCKmer(kmerList):
    rckmerList = set()
    myDict = {
        'A': 'T',
        'C': 'G',
        'G': 'C',
        'T': 'A'
    }
    for kmer in kmerList:
        rckmerList.add(sorted([kmer, ''.join([myDict[nc] for nc in kmer[::-1]])])[0])
    return sorted(rckmerList)

#########################################################################
def Kmer(fastas, k=2, upto=False, normalize=True):
    encoding = []
    header = ['#']
    NA = 'ACGT'
    if k < 1:
        print('The k must be positive integer.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            code = [name]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            kmers = kmerArray(sequence, k)
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = [name]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    return encoding



def RCKmer(fastas, k=2, upto=False, normalize=True):
    encoding = []
    header = ['#']
    NA = 'ACGT'

    if k < 1:
        print('The k must be positive integer.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            tmpHeader = []
            for kmer in itertools.product(NA, repeat=tmpK):
                tmpHeader.append(''.join(kmer))
            header = header + generateRCKmer(tmpHeader)
        myDict = {}
        for kmer in header[2:]:
            rckmer = RC(kmer)
            if kmer != rckmer:
                myDict[rckmer] = kmer
        encoding.append(header)
        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                for j in range(len(kmers)):
                    if kmers[j] in myDict:
                        kmers[j] = myDict[kmers[j]]
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            code = [name]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    else:
        tmpHeader = []
        for kmer in itertools.product(NA, repeat=k):
            tmpHeader.append(''.join(kmer))
        header = header + generateRCKmer(tmpHeader)
        myDict = {}
        for kmer in header[2:]:
            rckmer = RC(kmer)
            if kmer != rckmer:
                myDict[rckmer] = kmer

        encoding.append(header)
        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            kmers = kmerArray(sequence, k)
            for j in range(len(kmers)):
                if kmers[j] in myDict:
                    kmers[j] = myDict[kmers[j]]
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = [name]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    return encoding



def Psednc(input_data,lamada=3, w=0.05, k = 2, extra_phyche_index=None):

    phyche_value = get_phyche_value_psednc(extra_phyche_index)
    
    fastas=readDNAFasta(input_data)

    vector = make_pseknc_vector(fastas, lamada, w, k, phyche_value, theta_type=1)

    return vector

def Pseknc(input_data, k=3, lamada=1, w=0.5,extra_phyche_index=None):

    phyche_value = get_phyche_value_pseknc(input_data, extra_phyche_index)        
    fastas=readDNAFasta(input_data)  
    vector=make_old_pseknc_vector(fastas, lamada, w, k, phyche_value, theta_type=1)
    
    return vector


def DAC(input_data,k=2,lag=1, phyche_index=None, all_property=False, extra_phyche_index=None):

    phyche_value = acc_property(input_data, k, phyche_index, all_property, extra_phyche_index)
    
    fastas=readDNAFasta(input_data)
    
    vector=make_ac_vector(fastas, lag, phyche_value, k)
    
    return vector
def DCC(input_data,k=2,lag=1, phyche_index=None, all_property=False, extra_phyche_index=None):

    phyche_value = acc_property(input_data, k, phyche_index, all_property, extra_phyche_index)
    
    fastas=readDNAFasta(input_data)
    
    vector=make_cc_vector(fastas, lag, phyche_value, k)
    
    return vector

def DACC(input_data, k=2,lag=1,phyche_index=None, all_property=False, extra_phyche_index=None):

    phyche_value = acc_property(input_data, k, phyche_index, all_property, extra_phyche_index)
    fastas=readDNAFasta(input_data)
    zipped = list(zip(make_ac_vector(fastas, lag, phyche_value, k),
                     make_cc_vector(fastas, lag, phyche_value, k)))
    vector = [reduce(lambda x, y: x + y, e) for e in zipped]

    return vector

def TAC(input_data,k=3,lag=1, phyche_index=None, all_property=False, extra_phyche_index=None):

    phyche_value = acc_property(input_data, k, phyche_index, all_property, extra_phyche_index)
    
    fastas=readDNAFasta(input_data)
    
    vector=make_ac_vector(fastas, lag, phyche_value, k)
    
    return vector
def TCC(input_data,k=3,lag=1, phyche_index=None, all_property=False, extra_phyche_index=None):

    phyche_value = acc_property(input_data, k, phyche_index, all_property, extra_phyche_index)
    
    fastas=readDNAFasta(input_data)
    
    vector=make_cc_vector(fastas, lag, phyche_value, k)
    
    return vector

def TACC(input_data, k=3,lag=1,phyche_index=None, all_property=False, extra_phyche_index=None):

    phyche_value = acc_property(input_data, k, phyche_index, all_property, extra_phyche_index)
    fastas=readDNAFasta(input_data)
    zipped = list(zip(make_ac_vector(fastas, lag, phyche_value, k),
                     make_cc_vector(fastas, lag, phyche_value, k)))
    vector = [reduce(lambda x, y: x + y, e) for e in zipped]
    return vector

