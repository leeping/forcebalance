"""This file contains a documentation generating script. Doxygen
is used to do the actual generation, so these functions act primarily to
streamline the process and provide some customizations to the automatically
generated documents
"""

import os
import re
import shutil
import subprocess
from datetime import datetime

def add_tab(fnm):
    """Add tabs to html version of general documentation"""
    newfile = []
    installtag = ' class="current"' if fnm.split('/')[-1] == 'installation.html' else ''
    usagetag = ' class="current"' if fnm.split('/')[-1] == 'usage.html' else ''
    tutorialtag = ' class="current"' if fnm.split('/')[-1] == 'tutorial.html' else ''
    glossarytag = ' class="current"' if fnm.split('/')[-1] == 'glossary.html' else ''
    todotag = ' class="current"' if fnm.split('/')[-1] == 'todo.html' else ''
    
    for line in open(fnm):
        newfile.append(line)
        if re.match('.*a href="index\.html"',line):
            newfile.append('      <li%s><a href="installation.html"><span>Installation</span></a></li>\n' % installtag)
            newfile.append('      <li%s><a href="usage.html"><span>Usage</span></a></li>\n' % usagetag)
            newfile.append('      <li%s><a href="tutorial.html"><span>Tutorial</span></a></li>\n' % tutorialtag)
            newfile.append('      <li%s><a href="glossary.html"><span>Glossary</span></a></li>\n' % glossarytag)
            newfile.append('      <li><a href="api/roadmap.html"><span>API</span></a></li>\n')
    with open(fnm,'w') as f: f.writelines(newfile)

def redirect_main_tab(fnm):
    """Redirect the 'main page' tab of API documentation to integrate with general documentation"""
    newfile=[]
    for line in open(fnm):
        if re.match('.*a href="index\.html"',line):
            newfile.append('      <li><a href="../index.html"><span>Main Page</span></a></li>\n')
            newfile.append('      <li ')
            if re.match('.*roadmap\.html$', fnm): newfile.append('class="current"')
            newfile.append('><a href="roadmap.html"><span>Project Roadmap</span></a></li>\n')
        else: newfile.append(line)
    with open(fnm,'w') as f: f.writelines(newfile)


def parse_html():
    """Look for HTML files that need processing"""
    for fnm in os.listdir('./html/'):
        if re.match('.*\.html$',fnm):
            add_tab('./html/' + fnm)

    for fnm in os.listdir('./html/api/'):
        if re.match('.*\.html$',fnm):
            redirect_main_tab('./html/api/' + fnm)

def find_forcebalance():
    """try to find forcebalance location in standard python path"""
    forcebalance_dir=""
    try:
        import forcebalance
        forcebalance_dir = forcebalance.__path__[0]
    except:
        print "Unable to find forcebalance directory in PYTHON PATH (Is it installed?)"
        print "Try running forcebalance/setup.py or you can always set the INPUT directory"
        print "manually in api.cfg"
        exit()

    return forcebalance_dir

def find_doxypy():
    """Check if doxypy is in system path or else ask for location of doxypy.py"""
    doxypy_path=""
    try:
        # first check to see if doxypy is in system path
        if subprocess.call(["doxypy", "makedocumentation.py"],stdout=open(os.devnull)): raise OSError()
        doxypy_path="doxypy"
    except OSError: 
        doxypy_path=raw_input("Enter location of doxypy.py: ")
        if not os.path.exists(doxypy_path) or doxypy_path[-9:] != 'doxypy.py':
            print "Invalid path to doxypy"
            exit()
    return doxypy_path

def doxyconf():
    """Try setting values in doxygen config files to match local environment"""
    doxypy_path=""

    # read .cfg, make temp file to edit, replace original when done
    with open('doxygen.cfg', 'r') as fin:
        lines = fin.readlines()

    shutil.copy('doxygen.cfg', 'doxygen.cfg.tmp')

    # make sure FILTER_PATTERNS is set to use doxypy
    with open('doxygen.cfg.tmp', 'w') as fout:
        for line in lines:
            if line.startswith('FILTER_PATTERNS        =') and not re.match(".*doxypy.*", line):
                doxypy_path = find_doxypy()
                option = 'FILTER_PATTERNS        = "*.py=' + doxypy_path + '"\n'
                fout.write(option)
            else:
                fout.write(line)

    shutil.move('doxygen.cfg.tmp', 'doxygen.cfg')

    # same with api.cfg but also check INPUT flag is set to forcebalance location
    # use same doxypy location as in doxygen.cfg
    with open('api.cfg', 'r') as fin:
        lines = fin.readlines()

    shutil.copy('api.cfg','api.cfg.tmp')

    with open('api.cfg.tmp', 'w') as fout:
        for line in lines:
            if line.startswith('INPUT                  =') and not re.match(".*forcebalance.*|.*src.*", line):
                option = 'INPUT                  = api.dox ' + find_forcebalance() + '\n'
                fout.write(option)
            elif line.startswith('FILTER_PATTERNS        =') and not re.match(".*doxypy.*", line):
                option = 'FILTER_PATTERNS        = "*.py=' + doxypy_path + '"\n'
                fout.write(option)
            else:
                fout.write(line)
    
    shutil.move('api.cfg.tmp', 'api.cfg')

def generate_content():
    """Parse text files and concatenate into mainpage and api"""
    # generate pages to be included in general documentation
    mainpage=""
    mainpage+="/**\n\n\\mainpage\n\n"
    for fnm in ["introduction.txt", "installation.txt", "usage.txt", "tutorial.txt", "glossary.txt", "option_index.txt"]:
        page=open(fnm,'r')
        mainpage+=page.read()
    mainpage+="\n\\image latex ForceBalance.pdf \"Logo.\" height=10cm\n\n*/"

    # generate pages to be included in API documentation
    api=""
    api+="/**\n\n"
    for fnm in ["roadmap.txt"]:
        page=open(fnm,'r')
        api+=page.read()
    api+="\n\n*/"

    return (mainpage, api)

def generate_doc(logfile=os.devnull):
    """Run doxygen and compile generated latex into pdf, optionally writing output to a file
    @param logfile Name of the log file to write to (defaults to null)
    """
    with open(logfile,'w') as log:
        # generate general documentation
        print "Generating program documentation with Doxygen..."
        subprocess.call(['doxygen', 'doxygen.cfg'], stdout=log, stderr=subprocess.STDOUT)
        shutil.copy('Images/ForceBalance.pdf','latex/')
        print "Compiling pdf..."
        os.chdir('latex')
        subprocess.call(['make'], stdout=log, stderr=subprocess.STDOUT)
        os.chdir('..')
        shutil.copy('latex/refman.pdf', 'ForceBalance-Manual.pdf')

        # generate technical (API) documentation
        print "Generating API documentation..."
        subprocess.call(['doxygen', 'api.cfg'], stdout=log, stderr=subprocess.STDOUT)
        print "Compiling pdf..."
        os.chdir('latex')
        subprocess.call(['make'], stdout=log, stderr=subprocess.STDOUT)
        os.chdir('..')
        shutil.copy('latex/refman.pdf', 'ForceBalance-API.pdf')

if __name__ == '__main__':
    print "Collecting documents..."
    os.system("python make-option-index.py > option_index.txt")
    mainpage, api = generate_content()

    branch = subprocess.check_output(['git','status']).splitlines()[0][12:]
    
    print "Stashing uncommitted changes on '%s'..." % branch

    os.system("git stash")
    print "Switching to doc repository..."
    os.system("git checkout gh-pages")
    print "Checking Doxygen config files..."
    doxyconf()

    print "Generating docs with Doxygen..."
    with open('mainpage.dox','w') as f:
        f.write(mainpage)
    with open('api.dox','w') as f:
        f.write(api)
    generate_doc()
    print "Integrating HTML docs..."
    parse_html()

    print "Applying changes to local repository..."
    os.system('git add ./html *.pdf')
    os.system('git commit -m "Automatic documentation generation on %s"' % datetime.now().strftime("%m-%d-%Y %H:%M"))

    print "\n----Enter username/password to push changes to remote repository or Ctrl-C to abort\n"
    os.system('git push origin gh-pages')
    print

    print "Cleaning up..."
    os.system("rm -rf latex option_index.txt api.dox mainpage.dox")   # cleanup

    print "Returning to branch '%s'..." % branch
    os.system('git checkout %s' % branch)


    print "Loading master branch stash if necessary..."
    os.system('git stash pop')
    print "Documentation successfully generated"
