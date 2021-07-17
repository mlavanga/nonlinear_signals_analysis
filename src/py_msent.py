import numpy as np


# Variables globales
nb_scales     = 20
length_sample = 1000


def embed_seq(time_series, tau, embedding_dimension):
    """Build a set of embedding sequences from given time series `time_series`
    with lag `tau` and embedding dimension `embedding_dimension`.
    Let time_series = [x(1), x(2), ... , x(N)], then for each i such that
    1 < i <  N - (embedding_dimension - 1) * tau,
    we build an embedding sequence,
    Y(i) = [x(i), x(i + tau), ... , x(i + (embedding_dimension - 1) * tau)].
    All embedding sequences are placed in a matrix Y.
    Parameters
    ----------
    time_series
        list or numpy.ndarray
        a time series
    tau
        integer
        the lag or delay when building embedding sequence
    embedding_dimension
        integer
        the embedding dimension
    Returns
    -------
    Y
        2-embedding_dimension list
        embedding matrix built
    Examples
    ---------------
    >>> import pyeeg
    >>> a=range(0,9)
    >>> pyeeg.embed_seq(a,1,4)
    array([[0,  1,  2,  3],
           [1,  2,  3,  4],
           [2,  3,  4,  5],
           [3,  4,  5,  6],
           [4,  5,  6,  7],
           [5,  6,  7,  8]])
    >>> pyeeg.embed_seq(a,2,3)
    array([[0,  2,  4],
           [1,  3,  5],
           [2,  4,  6],
           [3,  5,  7],
           [4,  6,  8]])
    >>> pyeeg.embed_seq(a,4,1)
    array([[0],
           [1],
           [2],
           [3],
           [4],
           [5],
           [6],
           [7],
           [8]])
    """
    if not type(time_series) == np.ndarray:
        typed_time_series = np.asarray(time_series)
    else:
        typed_time_series = time_series

    shape = (
        typed_time_series.size - tau * (embedding_dimension - 1),
        embedding_dimension
    )

    strides = (typed_time_series.itemsize, tau * typed_time_series.itemsize)

    return np.lib.stride_tricks.as_strided(
        typed_time_series,
        shape=shape,
        strides=strides
    )

def in_range(Template, Scroll, Distance):
	"""Determines whether one vector is the the range of another vector.
	
	The two vectors should have equal length.
	
	Parameters
	-----------------
	Template
		list
		The template vector, one of two vectors being compared
	Scroll
		list
		The scroll vector, one of the two vectors being compared
		
	D
		float
		Two vectors match if their distance is less than D
		
	Bit
		
	
	Notes
	-------
	The distance between two vectors can be defined as Euclidean distance
	according to some publications.
	
	The two vector should of equal length
	
	"""
	
	for i in range(0,  len(Template)):
			if abs(Template[i] - Scroll[i]) > Distance:
			     return False
	return True
	""" Desperate code, but do not delete
	def bit_in_range(Index): 
		if abs(Scroll[Index] - Template[Bit]) <=  Distance : 
			print "Bit=", Bit, "Scroll[Index]", Scroll[Index], "Template[Bit]",\
			 Template[Bit], "abs(Scroll[Index] - Template[Bit])",\
			 abs(Scroll[Index] - Template[Bit])
			return Index + 1 # move 
	Match_No_Tail = range(0, len(Scroll) - 1) # except the last one 
#	print Match_No_Tail
	# first compare Template[:-2] and Scroll[:-2]
	for Bit in range(0, len(Template) - 1): # every bit of Template is in range of Scroll
		Match_No_Tail = filter(bit_in_range, Match_No_Tail)
		print Match_No_Tail
		
	# second and last, check whether Template[-1] is in range of Scroll and 
	#	Scroll[-1] in range of Template
	# 2.1 Check whether Template[-1] is in the range of Scroll
	Bit = - 1
	Match_All =  filter(bit_in_range, Match_No_Tail)
	
	# 2.2 Check whether Scroll[-1] is in the range of Template
	# I just write a  loop for this. 
	for i in Match_All:
		if abs(Scroll[-1] - Template[i] ) <= Distance:
			Match_All.remove(i)
	
	
	return len(Match_All), len(Match_No_Tail)
	"""

def samp_entropy(X, M, R):
    """Computer sample entropy (SampEn) of series X, specified by M and R.
    SampEn is very close to ApEn.
    
    Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
    embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of
    Em is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension
    are 1 and M-1 respectively. Such a matrix can be built by calling pyeeg
    function as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only
    difference with Em is that the length of each embedding sequence is M + 1
    
    Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elements
    are Em[i][k] and Em[j][k] respectively. The distance between Em[i] and
    Em[j] is defined as 1) the maximum difference of their corresponding scalar
    components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two
    1-D vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance
    between them is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the
    value of R is defined as 20% - 30% of standard deviation of X.
    
    Pick Em[i] as a template, for all j such that 0 < j < N - M , we can
    check whether Em[j] matches with Em[i]. Denote the number of Em[j],
    which is in the range of Em[i], as k[i], which is the i-th element of the
    vector k.
    
    We repeat the same process on Emp and obtained Cmp[i], 0 < i < N - M.
    The SampEn is defined as log(sum(Cm)/sum(Cmp))
    References
    ----------
    Costa M, Goldberger AL, Peng C-K, Multiscale entropy analysis of biological
    signals, Physical Review E, 71:021906, 2005
    See also
    --------
    ap_entropy: approximate entropy of a time series
    """

    N = len(X)

    Em = embed_seq(X, 1, M)
    A = np.tile(Em, (len(Em), 1, 1))
    B = np.transpose(A, [1, 0, 2])
    D = np.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
    InRange = np.max(D, axis=2) <= R
    np.fill_diagonal(InRange, 0)  # Don't count self-matches

    Cm = InRange.sum(axis=0)  # Probability that random M-sequences are in range
    Dp = np.abs(
        np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T
    )

    Cmp = np.logical_and(Dp <= R, InRange[:-1, :-1]).sum(axis=0)

    # Avoid taking log(0)
    Samp_En = np.log(np.sum(Cm + 1e-100) / np.sum(Cmp + 1e-100))

    return Samp_En


## Coarse graining procedure
# tau : scale factor
# signal : original signal
# return the coarse_graining signal
def coarse_graining(tau, signal):
    # signal lenght
    N = len(signal)
    # Coarse_graining signal initialisation
    y = np.zeros(int(len(signal) / tau))
    for j in range(0, int(N / tau)):
        y[j] = sum(signal[i] / tau for i in range(int((j - 1) * tau), int(j * tau)))
    return y


## Multi-scale entropy
# m : length of the patterns that compared to each other
# r : tolerance
# signal : original signal
# return the Multi-scale entropy of the original signal (array of nbscales length)
def mse(m, r, signal, nbscales=None):
    # Output initialisation
    if nbscales == None:
        nbscales = int((len(signal) * nb_scales) / length_sample)
    y = np.zeros(nbscales + 1)
    y[0] = float('nan')
    for i in range(1, nbscales + 1):
        y[i] = samp_entropy(coarse_graining(i, signal), m, r)
    return y


## calculation of the matching number
# it use in the refined composite multi-scale entropy calculation
def match(signal, m, r):
    N = len(signal)

    Em = embed_seq(signal, 1, m)
    Emp = embed_seq(signal, 1, m + 1)

    Cm, Cmp = np.zeros(N - m - 1) + 1e-100, np.zeros(N - m - 1) + 1e-100
    # in case there is 0 after counting. Log(0) is undefined.

    for i in range(0, N - m):
        for j in range(i + 1, N - m):  # no self-match
            # if max(abs(Em[i]-Em[j])) <= R:  # v 0.01_b_r1
            if in_range(Em[i], Em[j], r):
                Cm[i] += 1
                # if max(abs(Emp[i] - Emp[j])) <= R: # v 0.01_b_r1
                if abs(Emp[i][-1] - Emp[j][-1]) <= r:  # check last one
                    Cmp[i] += 1

    return sum(Cm), sum(Cmp)


## Refined Composite Multscale Entropy
# signal : original signal
# m : length of the patterns that compared to each other
# r : tolerance
# nbscales :
# return the RCMSE of the original signal (array of nbscales length)
def rcmse(signal, m, r, nbscales):
    Nm = 0
    Nmp = 0
    y = np.zeros(nbscales + 1)
    y[0] = float('nan')
    for i in range(1, nbscales + 1):
        for j in range(0, i):
            (Cm, Cmp) = match(coarse_graining(i, signal[i:]), m, r)
            Nm += Cm
            Nmp += Cmp
        y[i] = -np.log(Nmp / Nm)
    return y


## Caclulate the complexity index of the MSE (or RCMSE) of the original signal
# sig : RCMSE or MSE of the original signal
# inf : lower bound for the calcul
# sup : upper bound for the calcul
# return the complexity index value
def complexity_index(sig, low, upp):
    ci = sum(sig[low:upp])
    return ci


## Calculate the cross-sample entropy of 2 signals
# u : signal 1
# v : signal 2
# m : length of the patterns that compared to each other
# r : tolerance
# return the cross-sample entropy value
def cross_SampEn(u, v, m, r):
    B = 0.0
    A = 0.0
    if (len(u) != len(v)):
        raise Exception("Error : lenght of u different than lenght of v")
    N = len(u)
    for i in range(0, N - m):
        for j in range(0, N - m):
            B += cross_match(u[i:i + m], v[j:j + m], m, r) / (N - m)
            A += cross_match(u[i:i + m + 1], v[j:j + m + 1], m + 1, r) / (N - m)
    B /= N - m
    A /= N - m
    cse = -np.log(A / B)
    return cse


## calculation of the matching number
# it use in the cross-sample entropy calculation
def cross_match(signal1, signal2, m, r):
    # return 0 if not match and 1 if match
    d = []
    for k in range(0, m):
        d.append(np.abs(signal1[k] - signal2[k]))
    if max(d) <= r:
        return 1
    else:
        return 0
