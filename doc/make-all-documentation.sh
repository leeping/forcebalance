#!/bin/bash

# First generate the documentation.
doxygen doxygen.cfg

# Push the documentation to our website.
rsync -auvz --delete html/ /home/leeping/Dropbox/Public/ForceBalance_Doc/
