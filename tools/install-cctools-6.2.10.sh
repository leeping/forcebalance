#!/bin/bash

# Download latest version from website.
echo "Downloading source."
#cctools="cctools-6.2.10"
version="6.2.10"
cctools_src="cctools-$version-source"
rm -rf $cctools_src $cctools_src.tar*
wget http://www3.nd.edu/~ccl/software/files/$cctools_src.tar.gz
echo "Extracting archive."
tar xzf $cctools_src.tar.gz
cd $cctools_src

# Increase all sorts of timeouts.
sed -i s/"timeout = 5;"/"timeout = 7200;"/g work_queue/src/*.c
sed -i s/"timeout = 10;"/"timeout = 7200;"/g work_queue/src/*.c
sed -i s/"timeout = 15;"/"timeout = 7200;"/g work_queue/src/*.c
sed -i s/"foreman_transfer_timeout = 3600"/"foreman_transfer_timeout = 86400"/g work_queue/src/work_queue.c
sed -i s/"long_timeout = 3600"/"long_timeout = 86400"/g work_queue/src/work_queue.c

# Disable perl
# sed -i s/"config_perl_path=auto"/"config_perl_path=no"/g configure

# Disable globus
# sed -i s/"config_globus_path=auto"/"config_globus_path=no"/g configure

#---- 
# Provide install prefix for cctools as well as 
# locations of Swig and Python packages (i.e. the
# executable itself is inside the bin subdirectory).
#
# This is to ensure that we can call the correct 
# versions of Python and Swig since the version 
# installed for the OS might be too old.
#----
prefix=$HOME/opt/cctools
swgpath=$(dirname $(dirname $(which swig)))
pypath=$(dirname $(dirname $(which python)))

pyver=$(python -c 'import sys; s=sys.version_info; print("%i.%i" % (s[0],s[1]))')

# Create these directories if they don't exist.
mkdir -p $prefix
mkdir -p $swgpath
mkdir -p $pypath

if [ ! -d $prefix ] ; then
    echo "Warning: Installation directory $prefix does not exist."
    read
fi

if [ ! -f $swgpath/bin/swig ] ; then
    echo "Warning: $swgpath does not point to Swig."
    read
fi

if [ ! -f $pypath/bin/python ] ; then
    echo "Warning: $pypath does not point to Python."
    read
fi

#----
# The following assumes that cctools will be installed into $HOME/opt
# and Python lives in $HOME/local.
#----
# Configure, make, make install.
if [[ $pyver =~ "3." ]] ; then
    ./configure --prefix $prefix/$version --with-python-path no --with-perl-path no --with-globus-path no --with-python3-path $pypath --with-swig-path $swgpath
else
    ./configure --prefix $prefix/$version --with-python-path $pypath --with-perl-path no --with-globus-path no --with-python3-path no --with-swig-path $swgpath
fi
make && make install && cd work_queue && make install

#----
# Make symbolic link from installed version to plain "cctools" folder.
# This allows you to add $HOME/cctools/bin to your PATH.
#----
# cd $prefix
# rm -f cctools
# ln -s $cctools cctools
# cd cctools/bin
# for i in wq_submit_workers.common sge_submit_workers torque_submit_workers slurm_submit_workers ; do 
#     if [ -f $HOME/etc/work_queue/$i ] ; then
#         echo "Replacing $i with LP's custom version"
#         mv $i $i.bak
#         ln -s $HOME/etc/work_queue/$i .
#     fi
# done
# cd ../..

# Install Python module.
echo "Before installing Python module, removing these files"
echo "from $pypath/lib/python${pyver}/site-packages/:"
rm -fv $pypath/lib/python${pyver}/site-packages/*work_queue*
echo "Now copying new Python module files:"
cp -rv $prefix/$version/lib/python${pyver}/site-packages/*work_queue* $pypath/lib/python${pyver}/site-packages/
echo "Python module installed"
