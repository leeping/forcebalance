#!/usr/bin/env python

import os
import re

"""
Made obsolete in July 2020
def add_tab(fnm):
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
            newfile.append('      <li%s><a href="todo.html"><span>To-Do&#160;List</span></a></li>\n' % todotag)
    with open(fnm,'w') as f: f.writelines(newfile)
            
for pth, dnm, fnms in os.walk('./html/'):
    for fnm in fnms:
        if re.match('.*\.html$',fnm):
            add_tab(os.path.join(pth,fnm))
"""

menudata_js = """
var menudata={children:[
{text:"Main Page",url:"index.html"},
{text:"Installation",url:"installation.html"},
{text:"Usage",url:"usage.html"},
{text:"Tutorial",url:"tutorial.html"},
{text:"Glossary",url:"glossary.html"},
{text:"Roadmap",url:"todo.html"},
{text:"Files",url:"files.html",children:[
{text:"File List",url:"files.html"}]}]}
"""

with open(os.path.join('html', 'menudata.js'), 'w') as f:
    f.write(menudata_js)
