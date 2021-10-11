import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

from cpython cimport array
import array

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.double
np.import_array()
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.double_t DTYPE_t
ctypedef np.int_t DTYPE_INT_t

# cimport cython
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
def fix(np.ndarray[DTYPE_t, ndim=2] Y, np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B, double Similarity_threshold, double Pa, int Max_length):
    assert Y.dtype == DTYPE and A.dtype == DTYPE and B.dtype == DTYPE
    cdef int i, j, k, m, q, index, lvalue=0
    cdef DTYPE_t temp
    cdef int y = Y.shape[0]  # y: num of miRNAs, x: num of diseases
    cdef int x = Y.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] P = np.zeros([x, y], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] Network_disease = np.zeros([x, 4, max(x, y)], dtype=DTYPE)  # 0:disease 1:miRNA
    cdef np.ndarray[DTYPE_t, ndim=3] Network_miRNA = np.zeros([y, 4, max(x, y)], dtype=DTYPE)
    cdef list l = []
    cdef list ll = []
    cdef list lll = []
    cdef list ll1 = []
    # cdef array.array lll = array.array('l',[])
    cdef array.array trick = array.array('l',[])
    cdef array.array trick1 = array.array('l',[])
    cdef array.array trick2 = array.array('l',[])
    # constructed a heterogeneous graph consisting of three interlinked sub-graphs


    for i in range(x):
        index = 0
        for j in range(x):
            if B[i,j] > Similarity_threshold:
                Network_disease[i,0,index] = B[i,j]  # disease-disease  similarity
                Network_disease[i,1,index] = (j + 1)
                index += 1
        index = 0
        for k in range(y):
            if Y[k,i] == 1:
                Network_disease[i,2,index] = 1  # disease-miRNA similarity
                Network_disease[i,3,index] = (k + 1)
                index += 1

    for i in range(y):
        index = 0
        for j in range(y):
            if A[i, j] > Similarity_threshold:
                Network_miRNA[i,2,index] = A[i, j]  # miRNA-miRNA  similarity
                Network_miRNA[i,3,index] = (j + 1)
                index += 1
        index = 0
        for k in range(x):
            if Y[i, k] == 1:
                Network_miRNA[i,0,index] = 1  # disease-disease similarity
                Network_miRNA[i,1,index] = (k + 1)
                index += 1
                ####################################################################

    ####################################################################
    # It is a little complicated to explain the process.
    # For example, this part works for the first iteration

    # for i in range(x):  #+:disease -:miRNA
    for i in range(x):  # +:disease -:miRNA
        # print(i)
        # l = []
        # ll = []
        # lll = [i + 1]
        l.clear()
        ll.clear()
        lll.clear()
        lll.append(i+1)

        for k in range(max(y, x)):  # disease
            if Network_disease[i,0,k] == 0:
                break
            if k != i:
                lll.append(<int>(Network_disease[i,1,k]))
                ll.append(lll)
                lll = [i + 1]
        for k in range(max(y, x)):  # miRNA
            if Network_disease[i,2,k] == 0:
                break
            lll.append(-<int>(Network_disease[i,3,k]))
            ll.append(lll)
            lll = [i + 1]
            P[i, -<int>(ll[-1][-1]) - 1] += 1

        ####################################################################

        ####################################################################
        # This part worked for the rest iterations based on the selected maximun path length
        # for j in range(Max_length-1):



        for j in range(Max_length - 1):
            ll1 = []
            # ll1.clear()
            for k in range(len(ll)):
                # unit=ll[k]
                # lll = list(ll[k])
                trick = array.array('i',ll[k])
                len_trick = len(trick)

                if trick.data.as_ints[j + 1] > 0:  # disease
                    for m in range(max(y, x)):  # disease
                        if Network_disease[trick.data.as_ints[j + 1] - 1, 0, m] == 0:
                            break;
                        # if (Network_disease[<int>(trick.data.as_ints[j + 1] - 1), 1, m]) not in trick:
                        #     ll1.append(trick + array.array('i', [<int>Network_disease[<int>(trick.data.as_ints[j + 1] - 1), 1, m]]))
                        if not int_in(<int>Network_disease[trick.data.as_ints[j + 1] - 1, 1, m], trick, len_trick):
                            array.resize_smart(trick2, len_trick + 1)
                            copy_and_extend(trick2, trick, <int>Network_disease[trick.data.as_ints[j + 1] - 1, 1, m], len_trick + 1)
                            ll1.append(array.copy(trick2))

                            # ll1.append(trick + array.array('i', [<int>Network_disease[trick.data.as_ints[j + 1] - 1, 1, m]]))
                    for m in range(max(y, x)):  # miRNA
                        temp = 1
                        if Network_disease[trick.data.as_ints[j + 1] - 1, 2, m] == 0:
                            break;
                        # if (-(Network_disease[<int>(trick.data.as_ints[j + 1] - 1), 3, m])) not in trick: #TODO:?
                        #     ll1.append(trick + array.array('i', [-<int>Network_disease[<int>(trick.data.as_ints[j + 1] - 1), 3, m]]))

                        if not int_in(-<int>(Network_disease[trick.data.as_ints[j + 1] - 1, 3, m]), trick, len_trick):
                            array.resize_smart(trick1, len_trick + 1)
                            copy_and_extend(trick1, trick, -<int>Network_disease[trick.data.as_ints[j + 1] - 1, 3, m], len_trick + 1)
                            ll1.append(array.copy(trick1))

                            # ll1.append(trick + array.array('i', [-<int>Network_disease[trick.data.as_ints[j + 1] - 1, 3, m]]))
                            # trick1 = array.copy(ll1[-1])

                            # for q in range(j + 2):
                            #     if ll1[-1][q + 1] > 0 and ll1[-1][q] > 0:  # disease-disease
                            #         temp *= B[<int>(ll1[-1][q] - 1), <int>(ll1[-1][q + 1] - 1)]
                            #     elif ll1[-1][q + 1] < 0 and ll1[-1][q] > 0:  # disease-miRNA
                            #         temp *= 1
                            #     elif ll1[-1][q + 1] > 0 and ll1[-1][q] < 0:  # miRNA-disease
                            #         temp *= 1
                            #     elif ll1[-1][q + 1] < 0 and ll1[-1][q] < 0:  # miRNA-miRNA
                            #         temp *= A[-<int>(ll1[-1][q]) - 1, -<int>(ll1[-1][q + 1]) - 1]
                            # P[i, -<int>(ll1[-1][-1]) - 1] += (temp ** (Pa * (j + 2)))
                            for q in range(j + 2):
                                if trick1.data.as_ints[q + 1] > 0 and trick1.data.as_ints[q] > 0:  # disease-disease
                                    temp *= B[trick1.data.as_ints[q] - 1, trick1.data.as_ints[q + 1] - 1]
                                elif trick1.data.as_ints[q + 1] < 0 and trick1.data.as_ints[q] > 0:  # disease-miRNA
                                    temp *= 1
                                elif trick1.data.as_ints[q + 1] > 0 and trick1.data.as_ints[q] < 0:  # miRNA-disease
                                    temp *= 1
                                elif trick1.data.as_ints[q + 1] < 0 and trick1.data.as_ints[q] < 0:  # miRNA-miRNA
                                    temp *= A[-trick1.data.as_ints[q] - 1, -trick1.data.as_ints[q + 1] - 1]
                            P[i, -trick1.data.as_ints[len(trick1)-1] - 1] += (temp ** (Pa * (j + 2)))
                if trick.data.as_ints[j + 1] < 0:  # miRNA
                    for m in range(max(y, x)):  # disease
                        if Network_miRNA[-trick.data.as_ints[j + 1] - 1, 0, m] == 0:
                            break;
                        # if Network_miRNA[-<int>(trick.data.as_ints[j + 1]) - 1, 1, m] not in trick:
                        #     ll1.append(trick + array.array('i', [<int>Network_miRNA[-<int>(trick.data.as_ints[j + 1]) - 1, 1, m]]))
                        if not int_in(<int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 1, m], trick, len_trick):
                            array.resize_smart(trick2, len_trick + 1)
                            copy_and_extend(trick2, trick, <int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 1, m], len_trick + 1)
                            ll1.append(array.copy(trick2))

                            # ll1.append(trick + array.array('i', [<int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 1, m]]))
                    for m in range(max(y, x)):  # miRNA
                        temp = 1
                        if Network_miRNA[-trick.data.as_ints[j + 1] - 1, 2, m] == 0:
                            break;
                        # if (-Network_miRNA[-<int>(trick.data.as_ints[j + 1]) - 1, 3, m]) not in trick:
                        if not int_in(-<int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 3, m], trick, len_trick):
                            array.resize_smart(trick1, len_trick + 1)
                            copy_and_extend(trick1, trick, -<int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 3, m], len_trick + 1)
                            ll1.append(array.copy(trick1))

                            # ll1.append(trick + array.array('i', [-<int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 3, m]]))
                            # trick1 = array.copy(ll1[-1])
                            # for q in range(j + 2):
                            #     if ll1[-1][q + 1] > 0 and ll1[-1][q] > 0:  # disease-disease
                            #         temp *= B[<int>(ll1[-1][q] - 1), <int>(ll1[-1][q + 1] - 1)]
                            #     elif ll1[-1][q + 1] < 0 and ll1[-1][q] > 0:  # miRNA-disease
                            #         temp *= 1
                            #     elif ll1[-1][q + 1] > 0 and ll1[-1][q] < 0:  # miRNA-disease
                            #         temp *= 1
                            #     elif ll1[-1][q + 1] < 0 and ll1[-1][q] < 0:  # miRNA-miRNA
                            #         temp *= A[-<int>(ll1[-1][q]) - 1, -<int>(ll1[-1][q + 1]) - 1]
                            # # The end of each iterations need to be aggregated with the scores
                            # P[i][-<int>(ll1[-1][-1]) - 1] += (temp ** (Pa * (j + 2)))
                            for q in range(j + 2):
                                if trick1.data.as_ints[q + 1] > 0 and trick1.data.as_ints[q] > 0:  # disease-disease
                                    temp *= B[trick1.data.as_ints[q] - 1, trick1.data.as_ints[q + 1] - 1]
                                elif trick1.data.as_ints[q + 1] < 0 and trick1.data.as_ints[q] > 0:  # miRNA-disease
                                    temp *= 1
                                elif trick1.data.as_ints[q + 1] > 0 and trick1.data.as_ints[q] < 0:  # miRNA-disease
                                    temp *= 1
                                elif trick1.data.as_ints[q + 1] < 0 and trick1.data.as_ints[q] < 0:  # miRNA-miRNA
                                    temp *= A[-trick1.data.as_ints[q] - 1, -trick1.data.as_ints[q + 1] - 1]
                            # The end of each iterations need to be aggregated with the scores
                            P[i, -trick1.data.as_ints[len(trick1)-1] - 1] += (temp ** (Pa * (j + 2)))


            ll = ll1
    return P.T

cdef int int_in(int num, array.array A, int size):
    cdef int i
    for i in range(size):
        if A.data.as_ints[i] == num:
            return True
    return False

cdef void copy_and_extend(array.array self, array.array cparray, int exnum, int selflen):
    # assert len(self) = len(cparray) + 1
    cdef int i
    for i in range(selflen-1):
        self.data.as_ints[i] = cparray.data.as_ints[i]
    self.data.as_ints[selflen-1] = exnum
