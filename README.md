# pycpp_binding_refcode
Here is the workable pycpp binding code. Mainly want to deal with opencv + cuda + python binding, one successful example is in cuda_stereobp, but the syntax is not sync with opencv-cpp now. Need changes. Just in case I forget how to create these binding codes. [To be cont.]

---------------------------------
Before you build the example in ./cuda_stereobp, make sure you can run opencv + cuda in .cpp env.

---------------------------------

**Build**
`$cd ./cuda_stereobp`

`$ python3 setupStereoBP.py build_ext --inplace` 

or python2 by using `python` instead of `python3`

after that you can see .so in ./cuda_stereobp folder

**How to use it**

in your python code,

`import sys`

`sys.path.append(\path\to\stereoBP.so)`

`import StereoBP as bp`

Currently, the pybinding stereoBP function should be used like this: 

displ = StereoBP.StereoBP_compute_8U(dst11, dst22, 
                                    _ndisp = 240,
                                    _iters = 20,
                                    _levels = 3,
                                    _msg_type = cv2.CV_16S,
                                    _maxDataTerm = 25.0,
                                    _dataWeight = 0.1,
                                    _maxDiscTerm = 15.0,
                                    _discSingleJump = 1.0)
                                    
(I know the syntax is different than opencv-cpp, but I need time to explore :\ )

----------------------------------------------

**Small Goal**

I notice there is no python binding codes for opencv functions with the usage of cuda. And I would like to develop it step by step. If anyone who is interested in this, you can watch this repo or send me a PR :)

Everyone needs a small goal :) Haha.