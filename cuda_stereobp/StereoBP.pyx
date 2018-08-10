from libcpp cimport bool
from cpython.ref cimport PyObject
from libcpp.vector cimport vector

# References PyObject to OpenCV object conversion code borrowed from OpenCV's own conversion file, cv2.cpp
cdef extern from 'pyopencv_converter.cpp':
    cdef PyObject* pyopencv_from(const Mat& m)
    cdef bool pyopencv_to(PyObject* o, Mat& m)

cdef extern from 'opencv2/core/cvstd.hpp' namespace 'cv':
    cdef cppclass Ptr[T]:
        T element_type
        Ptr() except + 
        Ptr(Ptr*) except +
        T& operator* () # probably no exceptions

cdef extern from 'opencv2/cudastereo.hpp' namespace 'cv::cuda':
    cdef cppclass StereoBeliefPropagation:
        @staticmethod
        Ptr[StereoBeliefPropagation] createStereoBeliefPropagation(int ndisp, int iters, int levels, int msg_type) except+
        #Expected to see error here
        void compute(GpuMat left, GpuMat right, GpuMat disparity)
        void setNumIters(int iters)
        void setNumLevels(int levels)
        void setMaxDataTerm(double max_data_term)
        void setDataWeight(double data_weight)
        void setMaxDiscTerm(double max_disc_term)
        void setDiscSingleJump(double disc_single_jump)
        void setMsgType(int msg_type)

cdef extern from 'opencv2/cudastereo.hpp' namespace 'cv::cuda':
    Ptr[StereoBeliefPropagation] createStereoBeliefPropagation(int ndisp, int iters, int levels, int msg_type) except+

cdef extern from 'opencv2/imgproc.hpp' namespace 'cv':
    cdef enum InterpolationFlags:
        INTER_NEAREST = 0
    cdef enum ColorConversionCodes:
        COLOR_BGR2GRAY

cdef extern from 'opencv2/core/core.hpp':
    cdef int CV_8UC1
    cdef int CV_32FC1
    cdef int CV_16S

cdef extern from 'opencv2/core/core.hpp' namespace 'cv':
    cdef cppclass Size_[T]:
        Size_() except +
        Size_(T width, T height) except +
        T width
        T height
    ctypedef Size_[int] Size2i
    ctypedef Size2i Size
    cdef cppclass Scalar[T]:
        Scalar() except +
        Scalar(T v0) except +

cdef extern from 'opencv2/core/core.hpp' namespace 'cv':
    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int) except +
        void* data
        int rows
        int cols

cdef extern from 'opencv2/core/cuda.hpp' namespace 'cv::cuda':
    cdef cppclass GpuMat:
        GpuMat() except +
        void upload(Mat arr) except +
        void download(Mat dst) const
    cdef cppclass Stream:
        Stream() except +

import numpy as np  # Import Python functions, attributes, submodules of numpy
cimport numpy as np  # Import numpy C/C++ API
from cython.operator cimport dereference

np.import_array()

def StereoBP_compute_8U(np.ndarray[np.uint8_t, ndim=2] _left_view,
                               np.ndarray[np.uint8_t, ndim=2] _right_view,
                               int _ndisp,
                               int _iters,
                               int _levels,
                               int _msg_type,
                               float _maxDataTerm,
                               float _dataWeight,
                               float _maxDiscTerm,
                               float _discSingleJump):
    # Create GPU/device InputArray for left_view and right_view
    cdef Mat left_view_mat
    cdef GpuMat left_view_gpu
    cdef Mat right_view_mat
    cdef GpuMat right_view_gpu
    pyopencv_to(<PyObject*> _left_view, left_view_mat)
    left_view_gpu.upload(left_view_mat)
    pyopencv_to(<PyObject*> _right_view, right_view_mat)
    right_view_gpu.upload(right_view_mat)

    # Create BP matcher
    cdef Ptr[StereoBeliefPropagation] stereoBP_matcher = createStereoBeliefPropagation(_ndisp, _iters, _levels, _msg_type)
    dereference(stereoBP_matcher).setMaxDataTerm(_maxDataTerm)
    dereference(stereoBP_matcher).setDataWeight(_dataWeight)
    dereference(stereoBP_matcher).setMaxDiscTerm(_maxDiscTerm)
    dereference(stereoBP_matcher).setDiscSingleJump(_discSingleJump)

    # Compute disp img from BP matcher
    cdef Mat disp_mat
    cdef GpuMat disp_gpu = GpuMat()
    dereference(stereoBP_matcher).compute(left_view_gpu, right_view_gpu, disp_gpu)

    # Get result of disp img
    disp_gpu.download(disp_mat)
    cdef np.ndarray out = <np.ndarray> pyopencv_from(disp_mat)
    return out
