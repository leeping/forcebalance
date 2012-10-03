#!/usr/bin/env python

import os, sys

fin = open(sys.argv[1]).readlines()
template = open(sys.argv[2]).readlines()

def determine_format_string(numstr): # Returns something like "% 8.3f"
    spl = numstr.split('.')
    if len(spl) != 2:
        return Exception("There should be exactly one decimal place in the word %s" % numstr)
    decims = len(spl[1])
    strlen = len(spl[0]) + decims + 1
    formstr = "%"
    if numstr[0] == "-":
        strlen -= 1
        formstr += " "
    formstr += "%i.%if" % (strlen, decims)
    return formstr

for line_temp, line_data in zip(template, fin):
    stemp = line_temp.split()
    sdata = line_data.split()
    line_out = line_temp
    for wt, wd in zip(stemp, sdata):
        if wt != wd:
            line_out = line_out.replace(wt, determine_format_string(wt) % float(wd), 1)
    print line_out,

