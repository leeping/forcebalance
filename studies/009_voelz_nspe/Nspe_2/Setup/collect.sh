tar xjf structures.tar.bz2
for i in `ls structures/xyz/*.xyz | awk -F '.o' '{print $1}' | awk -F 'c.' '{print $NF}' | sort -n | uniq`; do
    for j in `ls structures/xyz/*.xyz | awk -F 'o.' '{print $NF}' | awk -F '.' '{print $1}' | sort -n | uniq`; do
	f=structures/xyz/min.c.$i.o.$j.xyz
	if [ -f $f ]; then
	    cat $f >> all.xyz
	fi
    done
done
