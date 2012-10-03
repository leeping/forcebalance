# Sourcing environment variables
# This is because I compiled GROMACS with Intel compilers
. /opt/intel/Compiler/11.1/072/bin/iccvars.sh intel64
. /opt/intel/Compiler/11.1/072/bin/ifortvars.sh intel64
# This is because the modified Gromacs requires us to turn off solvent
# optimization (no longer needed)
export GMX_NO_SOLV_OPT=""
