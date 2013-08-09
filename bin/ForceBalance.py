#!/usr/bin/env python

""" @package ForceBalance

Executable script for starting ForceBalance. """

import sys, re
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer
from forcebalance.nifty import printcool
from forcebalance.output import getLogger, RawStreamHandler, INFO
import numpy
numpy.seterr(all='raise')

def Run_ForceBalance(input_file):
    """ Create instances of ForceBalance components and run the optimizer.

    The triumvirate, trifecta, or trinity of components are:
    - The force field
    - The objective function
    - The optimizer
    Cipher: "All I gotta do here is pull this plug... and there you have to watch Apoc die"
    Apoc: "TRINITY" *chunk*

    The force field is a class defined in forcefield.py.
    The objective function is a combination of target classes and a penalty function class.
    The optimizer is a class defined in this file.
    """
    # Add logging handlers to print to console
    logger=getLogger("forcebalance")
    logger.addHandler(RawStreamHandler(sys.stdout))
    logger.setLevel(INFO)
    
    ## The general options and target options that come from parsing the input file
    options, tgt_opts = parse_inputs(input_file)
    ## The force field component of the project
    forcefield  = FF(options)
    ## The objective function
    objective   = Objective(options, tgt_opts, forcefield)
    ## The optimizer component of the project
    optimizer   = Optimizer(options, objective, forcefield)
    ## Actually run the optimizer.
    optimizer.Run()

def process(word, color):
    if color == 'black':
        Answer = word
    elif color == 'blue':
        Answer = "\x1b[44m" + " ".join(['' for i in range(len(word)+1)]) +  "\x1b[0m"
    elif color == 'gold':
        Answer = "\x1b[43m" + " ".join(['' for i in range(len(word)+1)]) +  "\x1b[0m"
    return Answer

def main():
    printcool("Welcome to ForceBalance version 1.1! =D\nForce Field Optimization System\nAuthor: Lee-Ping Wang", ansi="1", bold=True, minwidth=64)
    logostr = """
                          ,'+++                                        
                       ,++++++.      .:,,.                              
                    :+++++'`   `;    `,:::::.`                          
                 '+++++'    `'++++      `,::::,,`                       
             `;+++++:    ,+++++'.          `.:::::,`                    
          `++++++,    :+++++'`                 .,::::,`                 
       .+++++'.    ;+++++'                        `,,:::,`              
    :+++++'`   `;+++++:                              `,::::,.`          
   ++++;`   `++++++:               `.:+@@###@'          `,,::::.`       
    :    ,++++++.          ,;'+###############.             .,::,       
      :++++++`         +################':`                    .        
      +++;`              `.,,...####+.                                  
                                ,#####      +##.               +++   +++
 ,,,,                            #####      ######             +++   +++
 ,::,                ###'        ####'     :#####'             +++   +++
 ,::,                :####@      ####.    ,####'               +++   +++
 ,::,                 ######     ####    +###+                 +++   +++
 ,::,                  #####     ####   ###;                   +++   +++
 ,::,                   :##      ####  ++`                     +++   +++
 ,::,                            ####``..:;+##############+`   +++   +++
 ,::,             .,:;;'++##################################`  +++   +++
 ,::,    `############################++++''';;;;;;;;;;;'';    +++   +++
 ,::,      ,########':,.``       ####                          +++   +++
 ,::,                            ####                          +++   +++
 ,::,                            ####                          +++   +++
 ,::,                            ####                          +++   +++
 ,::,                            ####                          +++   +++
 ,::,                            ####                          +++   +++
 ,::,                            ####                          +++   +++
 ,,,,                            ####                          +++   +++
      ;++,                       ####                                   
     `'+++++:                    ####                                   
    `    '+++++;                 ####                       `.,:.       
   ++++,    :+++++'`             ####                    `,:::::.       
   .'+++++,    :++++++.          ###                  `,::::,.`         
      `'+++++;    .++++++,        +`               .,::::,`             
          ;+++++'`   `++++++:                   .,:::,,`                
             :+++++'.   `;+++++;`           `.:::::,`                   
                ,++++++`    '++++++      `,::::,,`                      
                   .'+++++:    ,+;    `,:::::.                          
                      `'+++++:       ,::,,.                             
                          ;++++.      ,`                                
                             ,`                                         
"""
    b = 'blue'
    g = 'gold'
    k = 'black'
    colorlist = [[],[b],[b,g],[b,b,g],[b,b,g],[b,b,g],[b,b,g],[b,b,g],
                 [b,b,g],[b,b,k,g],[b,b,k,g],[b,k,g],[b,k],[k,k,b,b],
                 [g,k,k,b,b],[g,k,k,k,b,b],[g,k,k,k,b,b],[g,k,k,k,b,b],
                 [g,k,k,k,b,b],[g,k,k,k,b,b],[g,k,b,b],[g,k,b,b],[g,k,b,b],
                 [g,k,k,b,b],[g,k,b,b],[g,k,b,b],[g,k,b,b],[g,k,b,b],[g,k,b,b],
                 [g,k,b,b],[g,k,b,b],[b,k],[b,k],[b,b,k,g],[b,b,k,g],[b,b,k,g],
                 [b,b,k,g],[b,b,g],[b,b,g],[b,b,g],[b,b,g],[b,g],[b,g],[b],[]]

    words = [l.split() for l in logostr.split('\n')]
    for ln, line in enumerate(logostr.split('\n')):
        # Reconstruct the line.
        words = line.split()
        whites = re.findall('[ ]+',line)
        newline = ''
        i = 0

        if len(line) > 0 and line[0] == ' ':
            while i < max(len(words), len(whites)):
                try:
                    newline += whites[i]
                except: pass
                try:
                    newline += process(words[i], colorlist[ln][i])
                except: pass
                i += 1
        elif len(line) > 0:
            while i < max(len(words), len(whites)):
                try:
                    newline += process(words[i], colorlist[ln][i])
                except: pass
                try:
                    newline += whites[i]
                except: pass
                i += 1
        print newline

    if len(sys.argv) != 2:
        print "Please call this program with only one argument - the name of the input file."
        sys.exit(1)
    Run_ForceBalance(sys.argv[1])

if __name__ == "__main__":
    main()
