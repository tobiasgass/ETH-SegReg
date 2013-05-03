
ADD_LIBRARY(ugmix
  bandmat.cpp   include.h     myexcept.h   newmat7.cpp   newmatnl.h    nm_misc.cpp   test_exc.cpp  tmt8.cpp  tmtf.cpp  tmtm.cpp
cholesky.cpp  jacobi.cpp    newfft.cpp   newmat8.cpp   newmatrc.h    precisio.h    tmt1.cpp      tmt9.cpp  tmtg.cpp  unsupervised.cpp
controlw.h    k_mean.cpp    newmat1.cpp  newmat9.cpp   newmatrm.cpp  sl_ex.cpp     tmt2.cpp      tmta.cpp  tmt.h     unsupervised.h
evalue.cpp    k_mean.h      newmat2.cpp  newmatap.h    newmatrm.h    solution.cpp  tmt3.cpp      tmtb.cpp  tmth.cpp
example.cpp         newmat3.cpp  newmatex.cpp  nl_ex.cpp     solution.h    tmt4.cpp      tmtc.cpp  tmti.cpp
fft.cpp       mem.cpp       newmat4.cpp  newmat.h      nm_ex1.cpp    sort.cpp      tmt5.cpp      tmt.cpp   tmtj.cpp
garch.cpp     mem.h         newmat5.cpp  newmatio.h    nm_ex2.cpp    submat.cpp    tmt6.cpp      tmtd.cpp  tmtk.cpp
hholder.cpp   myexcept.cpp  newmat6.cpp  newmatnl.cpp  nm_ex3.cpp    svd.cpp       tmt7.cpp      tmte.cpp  tmtl.cpp
)
IF (TCMALLOC)

TARGET_LINK_LIBRARIES (ugmix tcmalloc)
ENDIF(TCMALLOC)
