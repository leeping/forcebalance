#!/usr/bin/env python

"""
Python call graph generator

Code analysis utility to determine the global call graph. NOT really part of ForTune!

@bug unable to resolve things with the same name.
@author Jiahao Chen
@date 2010
"""

import glob, StringIO, sys, tokenize

class node:
    """
    Data structure for holding information about python objects
    """
    def __init__(self, nodeid = 0, name = '', parent = None, dtype = None):
        """
        Constructor
        """
        self.oid = nodeid
        self.name = name
        self.parent = parent
        self.dtype = dtype

def main():
    """
    """
    modulelist = []
    for filename in glob.glob('*.py'):
        module = filename[:-3]
        #Exclude self
        if module not in ['callgraph','nifty']:
            modulelist.append(module)
    
    
    
    knowntypeslist = ['def', 'class']
    
    ##Scans for classes and functions in modules
    globalobjectlist = {}
    ##Assigns each object unique id, even if names repeat
    objid = 0 
    for module in modulelist:
        level = 0
        
        obj = node(objid, module, None, 'file')
        objid += 1
    
        objstack = [(-1, obj)]
        objlist = [obj]
        for line in open(module+'.py'):
            t = line.split()
            if len(t) > 0:
                thislevel = line.find(t[0])
                #Keeps track of objid, name, datatype and indentation level
                objstack = [x for x in objstack if x[0] < thislevel]
                level = thislevel
    
                if t[0] in knowntypeslist:
                    thisname = t[1].split('(')[0].split(')')[0].split(':')[0]
                    thisobj = node(objid, thisname, objstack[-1][1].oid, t[0])
                    objlist.append(thisobj)
                    objstack.append((level, thisobj))
                    objid += 1
    
        globalobjectlist[module] = objlist
        
    #Scan for references in modules
    references = []
    for module in modulelist:
        thisobj = globalobjectlist[module][0]
        objstack = [(-1, thisobj)]
        level = 0
        #Initialize known namespace
        localnamespace = globalobjectlist[module]
    
        inimport = indeclar = False
    
        for line in open(module+'.py'):
            t = line.split()
            if len(t) > 0:
                thislevel = line.find(t[0])
                #Keeps track of objid, name, datatype and indentation level
                objstack = [x for x in objstack if x[0] < thislevel]
                level = thislevel
    
                if t[0] in knowntypeslist:
                    thisname = t[1].split('(')[0].split(')')[0].split(':')[0]
                    for obj in localnamespace:
                        if obj.name == thisname:
                            thisobj = obj
                            break
                    objstack.append((level, thisobj))
                    indeclar = (line[-2] == '\\')
                            
                if 'import' in t:
                    if 'import' in t[0]: #import _
                        pos = line.find('import') + len('import')
                    elif 'import' in t[2]: #from _ import _,_
                        pos = line.find('from') + len('from')
                    inimport = True
                    thatmodule = t[1]
    
                if inimport:
                    imported = line[pos:].replace(',',' ').split()
                   
                    if thatmodule in modulelist:
                        for thatname in imported:
                            for thatobj in globalobjectlist[thatmodule][::-1]:
                                if thatobj.name == thatname:
                                    localnamespace.append(thatobj)
                                    break
    
                    if line[-2] != '\\':
                        inimport = False
                    else:
                        pos = 0
    
                #Handles in a simple manner the case where a python file is called
                #in a shell system call
                #The assumption is that the call is of the form python _.py ...
    
                for thatmodule in modulelist:
                    if 'python '+thatmodule+'.py' in line and module != thatmodule:
                        references.append((thisobj, globalobjectlist[thatmodule][0]))
    
    
                if not indeclar and t[0] not in knowntypeslist:
                    try:
                        g = tokenize.generate_tokens(StringIO.StringIO(line).readline)
    
                        for _, token, _, _, _ in g:
                            matches = []
                            for someobj in localnamespace[::-1]:
                                thisobj = objstack[-1][1]
                                if token == someobj.name and someobj.oid != thisobj.oid:
                                    matches.append(someobj)
                            
                            if len(matches)>1:
                                print >> sys.stderr, "Multiple matches detected for", token
                                print >> sys.stderr, "!!! Graph may contain errors !!!"
                                for obj in matches:
                                    if obj.parent == thisobj.oid:
                                        match = obj
                                        break
                                #print >> sys.stderr, "Arbitrarily picking parent"
    
                                if [thisobj, match] not in references:
                                    references.append([thisobj, match])
                            elif len(matches) == 1:
                                match = matches[0]
                                if [thisobj, match] not in references:
                                    references.append([thisobj, match])
                        
                    except tokenize.TokenError: #breaks on multiline thingies
                        pass
                else:
                    indeclar = (line[-2] == '\\')
    
    
    
    print "digraph G {"
    
    #Print modules
    print 'subgraph {'
    #print '    rank = same'
    for module in modulelist:
        thisid = [obj.oid for obj in globalobjectlist[module] if \
                obj.name == module and obj.dtype == 'file'][0]
        print '    ', thisid, '[ label="'+module+'", color = grey, style="rounded,filled", fillcolor = yellow]'
    print '}'    
    
    #Print objects
    inherits = []
    for module in modulelist:
        for obj in globalobjectlist[module]:
            shape = None
            if obj.dtype == 'def':
                shape = 'box'
            elif obj.dtype == 'class':
                shape = 'house'
            if shape != None:
                print obj.oid, '[label = "'+obj.name+'", shape =', shape, ']'
                if (obj.oid, obj.parent) not in inherits:
                    inherits.append((obj.oid, obj.parent))
    
    for a, b in inherits:
        print a, '->', b, '[style = dashed, arrowhead = none]'
    
    print "#Call graph"
    
    for a, b in references:
        print a.oid, '->', b.oid
    print "}"
    
    #List orphans
    #import sys
    #for name in hasref:
    #    if not hasref[name]:
    #        print >> sys.stderr, name,' is not referenced by any other function'

if __name__ == "__main__":
    main()
