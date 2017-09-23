#!/bin/bash

# Pre-processing to build the main page document.
./make-option-index.py > option_index.txt

cat <<EOF > mainpage.py
"""

@mainpage

EOF

for i in "introduction.txt" "installation.txt" "usage.txt" "tutorial.txt" "glossary.txt" "option_index.txt"; do
    cat $i >> mainpage.py
    echo >> mainpage.py
done

cat <<EOF >> mainpage.py

\image latex ForceBalance.pdf "Logo." height=10cm

"""

EOF

echo \"\"\" >> mainpage.py

# Actually generate the documentation.
doxygen doxygen.cfg

# Post-processing to add tabs.
./add-tabs.py

# Push the documentation to our website.
#rsync -auvz --delete html/ /home/leeping/Dropbox/Public/ForceBalance_Doc/

# Go into the latex directory and make the PDF manual.
cd latex
make
cp refman.pdf ../ForceBalance-Manual.pdf
