import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
cimport numpy as np
import pandas as pd
cimport cython
from scipy import sparse


def load_fragments(filename):
    """Load fragments from a (possibly compressed) text file.

    Each line corresponds to a fragment.
    The expected format is:
        chromosome  start   end     id
    Where `id` is 0-based

    Parameters
    ----------
    filename : str
        Pile path

    Returns
    -------
    pd.DataFrame

    """
    return pd.read_table(filename, header=None, index_col=None, names=['chrom', 'pos1', 'pos2', 'id'])


cdef create_mat(df, unsigned int size, unsigned int k=0):
    """Create a Scipy sparse matrix from a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of Hi-C contacts
    size : int
        Size of the matrix
    k : int
        Number of diagonals to remove

    Returns
    -------
    scipy.sparse.csr_matrix

    """
    if k:
        # Remove diagonal (and off-diagonals) elements. k=1 is diagonal, k=2 is diagonal +/- 1, etc.
        df = df[(df['i']-df['j']).abs()>=k]

    row, col, data = df.as_matrix().T
    return sparse.coo_matrix((data, (row, col)), shape=(size, size), dtype=np.float64).tocsr()


cpdef load_contacts(filename, fragments, unsigned int diag=0, unsigned int chunksize=0):
    """Load contacts from a (possibly compressed) text file.

    Each line corresponds to a contact.
    The expected format is:
        rowID   colID   d
    Where `rowID` and `colID` are 0-based fragments IDs and `d` is the contact counts.

    Parameters
    ----------
    filename
    fragments
    diag
    chunksize

    Returns
    -------
    scipy.sparse.csr_matrix

    """
    cdef unsigned int size = len(fragments)

    if chunksize:
        mat = None
        reader = pd.read_table(filename, header=None, index_col=None, names=['i', 'j', 'v'], dtype=np.int32, chunksize=chunksize)

        for df in reader:
            if mat is None:
                mat = create_mat(df, size, k=diag)
            else:
                mat += create_mat(df, size, k=diag)
    else:
        df = pd.read_table(filename, header=None, index_col=None, names=['i', 'j', 'v'], dtype=np.int32)
        mat = create_mat(df, size, k=diag)

    # We expect an upper triangular matrix: make it symmetric
    mat += sparse.triu(mat, 1).T
    mat.sort_indices()
    return mat

class ModZScore:
    """

    Attributes
    ----------
    const : float

    median : float

    mad : float

    scores : np.array
    """
    def __init__(self, values):
        # MAD constant
        self.const = 0.6745
        # Median of values
        self.median = np.median(values)
        # Deviations
        deviations = values - self.median
        # Median absolute deviation
        self.mad = np.median(np.abs(deviations))
        self.scores = self.const * deviations / self.mad

    def get_scores(self):
        """

        Returns
        -------

        """
        return self.scores

    def calc_modzscore(self, value):
        """

        Parameters
        ----------
        value

        Returns
        -------

        """
        return self.const * (value - self.median) / self.mad


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef filter_low_bins(mat, double p):
    """

    Parameters
    ----------
    mat
    p

    Returns
    -------

    """
    cdef:
        np.ndarray[np.float64_t, ndim=1] bin_counts = np.array(mat.sum(axis=0)).flatten()
        double cutoff = np.percentile(bin_counts[bin_counts>0], p*100)

    clear_matrix(mat, (bin_counts < cutoff).astype(np.int32))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef filter_high_bins(mat, double p):
    """

    Parameters
    ----------
    mat
    p

    Returns
    -------

    """
    cdef:
        np.ndarray[np.float64_t, ndim=1] bin_counts = np.array(mat.sum(axis=0)).flatten()
        double cutoff = np.percentile(bin_counts[bin_counts>0], 100 - p * 100)

    clear_matrix(mat, (bin_counts > cutoff).astype(np.int32))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int is_local_minima(values, i, n=5):
    """

    Parameters
    ----------
    values
    i
    n

    Returns
    -------

    """
    if n < i + n < values.size and values[i-1] > values[i]:
        for j in np.arange(1, n+1):
            if values[i] > values[i+j]:
                return 0

        return 1
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef filter_auto(mat, output, use_last=False):
    """

    Parameters
    ----------
    mat
    output

    Returns
    -------

    """
    cdef:
        np.ndarray[np.float64_t, ndim=1] bin_counts = np.array(mat.sum(axis=0)).flatten()
        unsigned int local_minima = 0
        unsigned int low_mzs_cutoff = 0
        unsigned int high_mzs_cutoff = 0
        float pc_low_mzs = 0
        float pc_high_mzs = 0
        float pc_loc_minima = 0
        unsigned int i
        float score

    mzs = ModZScore(bin_counts)

    plt.style.use('ggplot')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("total counts per bin before correction")
    ax1.set_ylabel("frequency")
    hist, bin_edges, patches = ax1.hist(bin_counts[mzs.get_scores() < 5], 100)
    ymin, ymax = ax1.get_ylim()

    for i in np.arange(hist.size):
        score = mzs.calc_modzscore(bin_edges[i])

        if not low_mzs_cutoff and score > -3.5:
            low_mzs_cutoff = bin_edges[i]
        elif not high_mzs_cutoff and score >= 3.5:
            high_mzs_cutoff = bin_edges[i]

        if (not local_minima or use_last) and is_local_minima(hist, i, n=5):
            local_minima = bin_edges[i]

    if low_mzs_cutoff:
        pc_low_mzs = round(float( (bin_counts < low_mzs_cutoff).sum() ) / bin_counts.size, 2)
        ax1.vlines(low_mzs_cutoff, ymin, ymax, color='#E69F00', label='-3.5 mZ-score ({:.0f}%)'.format(pc_low_mzs*100))

    if high_mzs_cutoff:
        pc_high_mzs = round(float((bin_counts >= high_mzs_cutoff).sum()) / bin_counts.size, 3)
        ax1.vlines(high_mzs_cutoff, ymin, ymax, color='#56B4E9', label='3.5 mZ-score ({:.1f}%)'.format(pc_high_mzs*100))

    if local_minima:
        pc_loc_minima = round(float((bin_counts < local_minima).sum()) / bin_counts.size, 2)
        ax1.vlines(local_minima, ymin, ymax, color='#009E73', label='local minima ({:.0f}%)'.format(pc_loc_minima*100))

    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

    return pc_loc_minima, pc_high_mzs


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def clear_matrix(m, np.ndarray[np.int32_t, ndim=1] mask):
    """

    Parameters
    ----------
    m
    mask

    """
    cdef:
        # data array
        np.ndarray[np.float64_t, ndim=1] data = m.data
        # index array
        np.ndarray[np.int32_t, ndim=1] indices = m.indices
        # pointer array
        np.ndarray[np.int32_t, ndim=1] indptr = m.indptr
        unsigned int x, i, j

    """
    We want to zero out all rows/columns
    It's simple for columns since indices[i] tells the column of the i-th element of data
    So if indices[0] = 4, data[0] is on the 5th (since 0-based) column
    The data[i] has to be zero out if mask[indices[i]] is true

    But it's a bit more difficult for rows: we have to use indptr
    indptr[i] tells where the i-th columns begins.
    If indptr[0] = 0 and indptr[1] = 3, the first column starts at position 0 and runs until position 3 (exclusive)

    Beginning with i = 0 (first column), we have to increment i to always have data[n] within the column described by indptr[i]
    if n >= indptr[i+1] it means the (i+1)-th column starts before (or at) the n-th position, so we have to increment i
    Finally, we check if mask[i] is true
    """
    j = 0
    for x, i in enumerate(indices):
        while x >= indptr[j+1]:
            j += 1

        if mask[i] or mask[j]:
            data[x] = 0


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef np.ndarray ice_norm(mat, unsigned int max_iter=50, double eps=1e-4, verbose=False):
    """

    Parameters
    ----------
    mat
    max_iter
    eps
    verbose

    Returns
    -------

    """
    cdef:
        unsigned int i
        double variance
        np.ndarray[np.float64_t, ndim=1] total_biases

    if np.abs(mat - mat.T).sum() > eps:
        raise ValueError('Non-symmetric matrix')

    # use triangular form (faster)
    mat = sparse.triu(mat).tocsr()
    m, n = mat.shape

    # Set each element of the vector of total biases to 1
    total_biases = np.ones(m, dtype=np.float64)
    prev_dbias = np.zeros(m, dtype=np.float64)

    for i in np.arange(max_iter):
        # Calculate Si (sum of the matrix over all rows)
        # sum_ds = np.array(mat.sum(axis=0)).flatten()
        sum_ds = np.array(mat.sum(axis=0)).flatten() + np.array(mat.sum(axis=1)).flatten() - mat.diagonal()

        # Calculate additional vector of biases
        # Renormalize dBias by its mean value over non-zero bins to avoid numerical instabilities
        dbias = sum_ds / sum_ds[sum_ds!=0].mean()

        # Set zero values of dBias to 1 to avoid zero division
        dbias[dbias==0] = 1

        # Divide Wij by Dbi * dBj for all (i, j)
        update_matrix(mat, dbias)

        # Multiply total vector of biases by additional biases
        total_biases *= dbias

        variance = np.abs(prev_dbias - dbias).sum()  # use mean instead of sum?
        if variance < eps:
            if verbose:
                sys.stderr.write("Break at iteration {}\n".format(i+1))
            break
        elif verbose:
            sys.stderr.write("Iteration {}: {}\n".format(i+1, variance))

        prev_dbias = dbias.copy()

    return total_biases


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def update_matrix(mat, np.ndarray[np.float64_t, ndim=1] biases):
    """

    Parameters
    ----------
    mat
    biases
    """
    cdef:
        np.ndarray[np.float64_t, ndim=1] data = mat.data
        np.ndarray[np.int32_t, ndim=1] indices = mat.indices
        np.ndarray[np.int32_t, ndim=1] indptr = mat.indptr
        unsigned int x, i, j

    j = 0
    for x, i in enumerate(indices):
        while x >= indptr[j+1]:
            j += 1

        data[x] /= biases[i] * biases[j]


def triu(mat):
    return sparse.triu(mat)