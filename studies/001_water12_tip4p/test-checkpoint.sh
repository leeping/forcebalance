paste <(grep -A1 Step 01_bfgs_from_start.out | awk '/^ +[0-9]/') <(cat <(grep -A1 Step 02a_bfgs_writechk.out | awk '/^ +[0-9]/') <(grep -A1 Step 02b_bfgs_readchk.out | awk '/^ +[1-9]/')) 
#| awk '{p+=($5 - $11)**2} END {print p}'

paste <(grep -A1 Step 01_bfgs_from_start.out | awk '/^ +[0-9]/') <(cat <(grep -A1 Step 02a_bfgs_writechk.out | awk '/^ +[0-9]/') <(grep -A1 Step 02c_bfgs_readmvals.out | awk '/^ +[1-9]/'))
