import sys,re,os
from functools import reduce
from collections import Counter
import pandas as pd
import numpy as np
import itertools
from math import sqrt
from math import pow

ALPHABET='ACGU'



def readRNAFasta(file):
	if os.path.exists(file) == False:
		print('Error: "' + file + '" RNA sequence does not exist.')
		sys.exit(1)

	with open(file) as f:
		records = f.read()

	if re.search('>', records) == None:
		print('The input RNA sequence must be fasta format.')
		sys.exit(1)

	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ACGU-]', '-', ''.join(array[1:]).upper())
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
def convert_phyche_index_to_dict(phyche_index):

    len_index_value = len(phyche_index[0])
    k = 0
    for i in range(1, 10):
        if len_index_value < 4**i:
            error_infor = 'Sorry, the number of each index value is must be 4^k.'
            sys.stdout.write(error_infor)
            sys.exit(0)
        if len_index_value == 4**i:
            k = i
            break
    from nacutil import make_kmer_list
    kmer_list = make_kmer_list(k, ALPHABET)
    # print kmer_list
    len_kmer = len(kmer_list)
    phyche_index_dict = {}
    for kmer in kmer_list:
        phyche_index_dict[kmer] = []
    # print phyche_index_dict
    phyche_index = list(zip(*phyche_index))
    for i in range(len_kmer):
        phyche_index_dict[kmer_list[i]] = list(phyche_index[i])

    return phyche_index_dict
def standard_deviation(value_list):


    n = len(value_list)
    average_value = sum(value_list) * 1.0 / n
    return sqrt(sum([pow(e - average_value, 2) for e in value_list]) * 1.0 / (n - 1))


def normalize_index(phyche_index, is_convert_dict=False):

    normalize_phyche_value = []
    for phyche_value in phyche_index:
        average_phyche_value = sum(phyche_value) * 1.0 / len(phyche_value)
        sd_phyche = standard_deviation(phyche_value)
        normalize_phyche_value.append([round((e - average_phyche_value) / sd_phyche, 2) for e in phyche_value])

    if is_convert_dict is True:
        return convert_phyche_index_to_dict(normalize_phyche_value)

    return normalize_phyche_value
def parallel_cor_function(nucleotide1, nucleotide2, phyche_index):
    """Get the cFactor.(Type1)"""
    temp_sum = 0.0
    phyche_index_values = list(phyche_index.values())
    len_phyche_index = len(phyche_index_values[0])
    for u in range(len_phyche_index):
        temp_sum += pow(float(phyche_index[nucleotide1][u]) - float(phyche_index[nucleotide2][u]), 2)

    return temp_sum / len_phyche_index


def series_cor_function(nucleotide1, nucleotide2, big_lamada, phyche_value):
    """Get the series correlation Factor(Type 2)."""
    return float(phyche_value[nucleotide1][big_lamada]) * float(phyche_value[nucleotide2][big_lamada])


def get_parallel_factor(k, lamada, sequence, phyche_value):
    """Get the corresponding factor theta list."""
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
    """Get the corresponding series factor theta list."""
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
    """Generate the pseknc vector."""
    kmer = make_kmer_list(k, ALPHABET)
    vector = []

    for sequence in sequence_list:
        if len(sequence) < k or lamada + k > len(sequence):
            error_info = "Sorry, the sequence length must be larger than " + str(lamada + k)
            sys.stderr.write(error_info)
            sys.exit(0)

        # Get the nucleotide frequency in the DNA sequence.
        fre_list = [frequency(sequence, str(key)) for key in kmer]
        fre_sum = float(sum(fre_list))

        # Get the normalized occurrence frequency of nucleotide in the RNA sequence.
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
    NA = 'ACGU'
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
    NA = 'ACGU'

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

phy=pd.read_csv('phy.csv',header=-1,index_col=None)
phyche_index=np.array(phy)
phyche_index_dict=normalize_index(phyche_index, is_convert_dict=True)

def Psednc(input_data,lamada=3, w=0.05, k = 2):

    phyche_value = phyche_index_dict
    
    fastas=readRNAFasta(input_data)

    vector = make_pseknc_vector(fastas, lamada, w, k, phyche_value, theta_type=1)

    return vector



def DAC(input_data,k=2,lag=1, phyche_index=None):

    phyche_value = phyche_index_dict
    
    fastas=readRNAFasta(input_data)
    
    vector=make_ac_vector(fastas, lag, phyche_value, k)
    
    return vector
def DCC(input_data,k=2,lag=1, phyche_index=None):

    phyche_value = phyche_index_dict
    
    fastas=readRNAFasta(input_data)
    
    vector=make_cc_vector(fastas, lag, phyche_value, k)
    
    return vector

def DACC(input_data, k=2,lag=1,phyche_index=None):

    phyche_value = phyche_index_dict
    fastas=readRNAFasta(input_data)
    zipped = list(zip(make_ac_vector(fastas, lag, phyche_value, k),
                     make_cc_vector(fastas, lag, phyche_value, k)))
    vector = [reduce(lambda x, y: x + y, e) for e in zipped]

    return vector
def NAC(fastas):
    NA =  'ACGU'
    encodings = []
    header = ['#']
    for i in NA:
        header.append(i)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = [name]
        for na in NA:
            code.append(count[na])
        encodings.append(code)
    return encodings
def DNC(fastas):
    base = 'ACGU'

    encodings = []
    dinucleotides = [n1 + n2 for n1 in base for n2 in base]
    header = ['#'] + dinucleotides
    encodings.append(header)

    AADict = {}
    for i in range(len(base)):
        AADict[base[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        tmpCode = [0] * 16
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings
def TNC(fastas):
    AA = 'ACGU'
    encodings = []
    triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
    header = ['#'] + triPeptides
    encodings.append(header)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        tmpCode = [0] * 64
        for j in range(len(sequence) - 3 + 1):
            tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j+1]]*4 + AADict[sequence[j+2]]] = tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j+1]]*4 + AADict[sequence[j+2]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings
def zCurve(x):
    t=[]
    A = x.count('A'); C = x.count('C'); G = x.count('G');TU=x.count('U')
    x_ = (A + G) - (C + TU)
    y_ = (A + C) - (G + TU)
    z_ = (A + TU) - (C + G)
            # print(x_, end=','); print(y_, end=','); print(z_, end=',')
    t.append(x_); t.append(y_); t.append(z_)
    return t  
def kmers(seq, k):
    v = []
    for i in range(len(seq) - k + 1):
        v.append(seq[i:i + k])
    return v
def monoMonoKGap(x, g):  # 1___1
    t=[]
    m = list(itertools.product(ALPHABET, repeat=2))
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 2)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-1] == gGap[1]:
                    C += 1
            t.append(C)
    return t
def monoDiKGap(x, g):  # 1___2    
    t=[]
    m = list(itertools.product(ALPHABET, repeat=3))
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 3)
        # seqLength = len(x) - (i+2) + 1
        # print(V)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-2] == gGap[1] and v[-1] == gGap[2]:
                    C += 1
                # print(C, end=',')
            t.append(C) 
    return t 
# find the path
Script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# choose the method
option = sys.argv[1]

# the input sequence
fastas = sys.argv[2]

 
if(option == "1"):
    #kmer method
     vector=Kmer(fastas, k=2)
     vector.to_csv("Kmer_out.csv")
elif(option == "2"):
    #RCKmer method
    vector=RCKmer(fastas, k=2)
    vector.to_csv("RCKKmer_out.csv")
elif(option == "3"):
    #Psednc method
    vector=Psednc(fastas,lamada=3, w=0.05, k = 2)
    vector.to_csv("Psednc_out.csv")
elif(option == "4"):
    #DAC method
    vector=DAC(fastas,k=2,lag=1)
    vector.to_csv("DAC_out.csv")
elif(option == "5"):
    #DCC method
    vector=DCC(fastas,k=2,lag=1)
    vector.to_csv("DCC_out.csv")
elif(option == "6"):
    #DACC method
    vector=DACC(fastas,k=2,lag=1)
    vector.to_csv("DACC_out.csv")
elif(option == "7"):
    #NAC method
    vector=NAC(fastas)
    vector.to_csv("NAC_out.csv")
elif(option == "8"):
    #DNC method
    vector=DNC(fastas)
    vector.to_csv("DNC_out.csv")
elif(option == "9"):
    #TNC method
    vector=TNC(fastas)
    vector.to_csv("TNC_out.csv")
elif(option == "10"):
    #zCurve method
    vector=zCurve(fastas)
    vector.to_csv("zCurve_out.csv")
elif(option == "11"):
    #monoMonoKGap method
    vector=monoMonoKGap(fastas, g=1)
    vector.to_csv("monoMonoKGap_out.csv")
elif(option == "12"):
    #monoDiKGap method
    vector=monoDiKGap(fastas, g=1)
    vector.to_csv("monoDiKGap_out.csv")
else:
    print("Invalid method number. Please check the method table!")



