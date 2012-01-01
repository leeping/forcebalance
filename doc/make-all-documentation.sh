#!/bin/bash

# First generate the documentation.
doxygen doxygen.cfg

# Post-processing to add tabs.
./add-tabs.py

# Push the documentation to our website.
rsync -auvz --delete html/ /home/leeping/Dropbox/Public/ForceBalance_Doc/

# Go into the latex directory and make the PDF manual.
cd latex
make
cp refman.pdf ..
cp refman.pdf /home/leeping/Dropbox/Public/ForceBalance_Doc/ForceBalance-Manual.pdf
