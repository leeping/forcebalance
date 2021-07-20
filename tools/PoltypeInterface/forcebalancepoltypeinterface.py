import os, numpy as np
import sys
import csv
import getopt
import shutil
import openbabel
from rdkit.Chem import rdmolfiles
from rdkit.Chem import Descriptors
import re
import numpy as np
from rdkit import Chem
import subprocess
import itertools 

poltypepathlist=None
vdwtypeslist=None
temperature_list=None
pressure_list=None
enthalpy_of_vaporization_list=None
enthalpy_of_vaporization_err_list=None
surface_tension_list=None
surface_tension_err_list=None
relative_permittivity_list=None
relative_permittivity_err_list=None
isothermal_compressibility_list=None
isothermal_compressibility_err_list=None
isobaric_coefficient_of_volume_expansion_list=None
isobaric_coefficient_of_volume_expansion_err_list=None
heat_capacity_at_constant_pressure_list=None
heat_capacity_at_constant_pressure_err_list=None
density_list=None
density_err_list=None
citation_list=None
fittypestogether=None

liquid_equ_steps=10000
liquid_prod_steps=5000000
liquid_timestep=1.0
liquid_interval=0.1
gas_equ_steps=500000
gas_prod_steps=1000000
gas_timestep=1.0
gas_interval=0.1
md_threads=4
liquid_prod_time=.1 #ns
gas_prod_time=1 #ns
nvtprops=False # NPT for most props, surface tension requires NVT and force balance complains
addwaterprms=True

def ReturnListOfList(string):
    newlist=string.split(',')
    templist=[]
    for ele in newlist:
        nums=ele.lstrip().rstrip().split()
        temp=[]
        for e in nums:
            temp.append(int(e))
        templist.append(temp)
    newlist=templist
    return newlist

temp=open(os.getcwd()+r'/'+'forcebalancepoltype.ini','r')
results=temp.readlines()
temp.close()
for line in results:
    if '#' not in line and line!='\n':
        if '=' in line:
            linesplit=line.split('=',1)
            a=linesplit[1].replace('\n','').rstrip().lstrip()
            newline=linesplit[0]
            if a=='None':
                continue
        else:
            newline=line

    if  "poltypepathlist" in newline:
        poltypepathlist=a.split(',')
    elif "fittypestogether" in newline:
        fittypestogether=ReturnListOfList(a)
    elif "vdwtypeslist" in newline:
        vdwtypeslist=ReturnListOfList(a)
    elif "citation_list" in newline:
        citation_list=ReturnListOfList(a)
    elif "temperature_list" in newline:
        temperature_list=ReturnListOfList(a)
    elif "pressure_list" in newline:
        pressure_list=ReturnListOfList(a)
    elif "enthalpy_of_vaporization_list" in newline:
        enthalpy_of_vaporization_list=ReturnListOfList(a)
    elif "enthalpy_of_vaporization_err_list" in newline:
        enthalpy_of_vaporization_err_list=ReturnListOfList(a)
    elif "surface_tension_list" in newline:
        surface_tension_list=ReturnListOfList(a)
    elif "surface_tension_err_list" in newline:
        surface_tension_err_list=ReturnListOfList(a)
    elif "relative_permittivity_list" in newline:
        relative_permittivity_list=ReturnListOfList(a)
    elif "relative_permittivity_err_list" in newline:
        relative_permittivity_err_list=ReturnListOfList(a)
    elif "isothermal_compressibility_list" in newline:
        isothermal_compressibility_list=ReturnListOfList(a)
    elif "isothermal_compressibility_err_list" in newline:
        isothermal_compressibility_err_list=ReturnListOfList(a)
    elif "isobaric_coefficient_of_volume_expansion_list" in newline:
        isobaric_coefficient_of_volume_expansion_list=ReturnListOfList(a)
    elif "isobaric_coefficient_of_volume_expansion_err_list" in newline:
        isobaric_coefficient_of_volume_expansion_err_list=ReturnListOfList(a)
    elif "heat_capacity_at_constant_pressure_list" in newline:
        heat_capacity_at_constant_pressure_list=ReturnListOfList(a)
    elif "heat_capacity_at_constant_pressure_err_list" in newline:
        heat_capacity_at_constant_pressure_err_list=ReturnListOfList(a)
    elif "density_list" in newline:
        density_list=ReturnListOfList(a)
    elif "density_err_list" in newline:
        density_err_list=ReturnListOfList(a)
    elif "liquid_equ_steps" in newline:
        liquid_equ_steps=int(a)
    elif 'liquid_prod_steps' in newline:
        liquid_prod_steps=int(a) 
    elif 'liquid_timestep' in newline:
        liquid_timestep=int(a)
    elif 'liquid_interval' in newline:
        liquid_interval=float(a)
    elif 'gas_equ_steps' in newline:
        gas_equ_steps=int(a)
    elif 'gas_prod_steps' in newline:
        gas_prod_steps=int(a)
    elif 'gas_timestep' in newline:
        gas_timestep=float(a)
    elif 'gas_interval' in newline:
        gas_interval=float(a)
    elif 'md_threads' in newline:
        md_threads=int(a)
    elif 'liquid_prod_time' in newline:
        liquid_prod_time=float(a)
    elif 'gas_prod_time' in newline:
        gas_prod_time=float(a)
    elif 'nvtprops' in newline:
        nvtprops=True

if temperature_list==None:
    raise ValueError('No temperature data')
if pressure_list==None:
    raise ValueError('No pressure data')
if density_list==None:
    raise ValueError('No density data')

def ShapeOfArray(array):
    dimlist=[]
    rows=len(array)
    dimlist.append(rows)
    for i in range(len(array)):
        row=array[i]
        cols=len(row)
        dimlist.append(cols)
    return dimlist

def CheckInputShapes(listofarrays):
    combs=itertools.combinations(listofarrays, 2)
    for comb in combs:
        arr1=comb[0]
        arr2=comb[1]
        shape1=ShapeOfArray(arr1)
        shape2=ShapeOfArray(arr2)
        if shape1!=shape2:
            raise ValueError('Input dimensions for Temperature, Pressure or Density are not the same') 

def GenerateNoneList(targetshape):
    ls=[]
    rows=targetshape[0]
    cols=targetshape[1:]
    for i in range(rows):
        colnum=cols[i]
        newls=[]
        for j in range(colnum):
            newls.append(None)
        ls.append(newls)
    return ls 



CheckInputShapes([temperature_list,pressure_list,density_list])
numberofmolecules=len(poltypepathlist)
targetshape=ShapeOfArray(temperature_list)
if enthalpy_of_vaporization_list==None:
    enthalpy_of_vaporization_list=GenerateNoneList(targetshape)

if enthalpy_of_vaporization_err_list==None:
    enthalpy_of_vaporization_err_list=GenerateNoneList(targetshape)

if surface_tension_list==None:
    surface_tension_list=GenerateNoneList(targetshape)

if surface_tension_err_list==None:
    surface_tension_err_list=GenerateNoneList(targetshape)

if relative_permittivity_list==None:
    relative_permittivity_list=GenerateNoneList(targetshape)

if relative_permittivity_err_list==None:
    relative_permittivity_err_list=GenerateNoneList(targetshape)

if isothermal_compressibility_list==None:
    isothermal_compressibility_list=GenerateNoneList(targetshape)

if isothermal_compressibility_err_list==None:
    isothermal_compressibility_err_list=GenerateNoneList(targetshape)

if isobaric_coefficient_of_volume_expansion_list==None:
    isobaric_coefficient_of_volume_expansion_list=GenerateNoneList(targetshape)

if isobaric_coefficient_of_volume_expansion_err_list==None:
    isobaric_coefficient_of_volume_expansion_err_list=GenerateNoneList(targetshape)

if heat_capacity_at_constant_pressure_list==None:
    heat_capacity_at_constant_pressure_list=GenerateNoneList(targetshape)

if heat_capacity_at_constant_pressure_err_list==None:
    heat_capacity_at_constant_pressure_err_list=GenerateNoneList(targetshape)

if density_list==None:
    density_list=GenerateNoneList(targetshape)

if density_err_list==None:
    density_err_list=GenerateNoneList(targetshape)

if citation_list==None:
    citation_list=GenerateNoneList(targetshape)


gas_prod_steps=int(1000000*gas_prod_time/gas_timestep)
liquid_prod_steps=int(1000000*liquid_prod_time/liquid_timestep)



def CombineData(temperature_list,pressure_list,enthalpy_of_vaporization_list,enthalpy_of_vaporization_err_list,surface_tension_list,surface_tension_err_list,relative_permittivity_list,relative_permittivity_err_list,isothermal_compressibility_list,isothermal_compressibility_err_list,isobaric_coefficient_of_volume_expansion_list,isobaric_coefficient_of_volume_expansion_err_list,heat_capacity_at_constant_pressure_list,heat_capacity_at_constant_pressure_err_list,density_list,density_err_list,citation_list):
    listoftptopropdics=[]
    for i in range(len(temperature_list)):
        temperatures=temperature_list[i]
        pressures=pressure_list[i]
        enthalpies=enthalpy_of_vaporization_list[i]
        enthalpies_err=enthalpy_of_vaporization_err_list[i]
        surf_tens=surface_tension_list[i]
        surf_tens_err=surface_tension_err_list[i]
        perms=relative_permittivity_list[i]
        perms_err=relative_permittivity_err_list[i]
        compress=isothermal_compressibility_list[i]
        compress_err=isothermal_compressibility_err_list[i]
        volumeexp=isobaric_coefficient_of_volume_expansion_list[i]
        volumeexp_err=isobaric_coefficient_of_volume_expansion_err_list[i]
        heatcap=heat_capacity_at_constant_pressure_list[i]
        heatcap_err=heat_capacity_at_constant_pressure_err_list[i]
        densities=density_list[i]
        densities_err=density_err_list[i]
        citations=citation_list[i]
        tptoproptovalue={}
        for j in range(len(temperatures)):
            temp=temperatures[j]
            pressure=pressures[j]
            tp=tuple([temp,pressure])
            tptoproptovalue[tp]={}   
            enthalpy=enthalpies[j]
            enthalpy_err=enthalpies_err[j]
            surf_ten=surf_tens[j] 
            surf_ten_err=surf_tens_err[j]
            perm=perms[j]
            perm_err=perms_err[j]
            comp=compress[j]
            comp_err=compress_err[j]
            volexp=volumeexp[j]
            volexp_err=volumeexp_err[j]
            cap=heatcap[j]
            cap_err=heatcap_err[j]
            density=densities[j]
            density_err=densities_err[j]
            citation=citations[j]
            tptoproptovalue[tp]['T']=temp
            tptoproptovalue[tp]['P']=pressure
            tptoproptovalue[tp]['Hvap']=enthalpy
            tptoproptovalue[tp]['Hvap_err']=enthalpy_err
            tptoproptovalue[tp]['Surf_Ten']=surf_ten
            tptoproptovalue[tp]['Surf_Ten_err']=surf_ten_err
            tptoproptovalue[tp]['Eps0']=perm
            tptoproptovalue[tp]['Eps0_err']=perm_err
            tptoproptovalue[tp]['Kappa']=comp
            tptoproptovalue[tp]['Kappa_err']=comp_err
            tptoproptovalue[tp]['Alpha']=volexp
            tptoproptovalue[tp]['Alpha_err']=volexp_err
            tptoproptovalue[tp]['Cp']=cap
            tptoproptovalue[tp]['Cp_err']=cap_err
            tptoproptovalue[tp]['Rho']=density
            tptoproptovalue[tp]['Rho_err']=density_err
            tptoproptovalue[tp]['cite']=citation

        listoftptopropdics.append(tptoproptovalue)
    return listoftptopropdics

listoftptopropdics=CombineData(temperature_list,pressure_list,enthalpy_of_vaporization_list,enthalpy_of_vaporization_err_list,surface_tension_list,surface_tension_err_list,relative_permittivity_list,relative_permittivity_err_list,isothermal_compressibility_list,isothermal_compressibility_err_list,isobaric_coefficient_of_volume_expansion_list,isobaric_coefficient_of_volume_expansion_err_list,heat_capacity_at_constant_pressure_list,heat_capacity_at_constant_pressure_err_list,density_list,density_err_list,citation_list)

       
def AddDefaultValues(tptoproptovalue):
    for tp,proptovalue in tptoproptovalue.items():
        globaldic={}
        defaultdenomvalue=1
        wtkeywordtovalue={}
        
        for prop,value in proptovalue.items():
            if prop!='name' and prop!='cite' and 'err' not in prop and prop!='T' and prop!='P':
                lowerprop=prop.lower()
                keyword=lowerprop+'_denom'
                globaldic[keyword]=defaultdenomvalue
                weightkeyword=prop+'_wt'
                try:
                    length=float(value)
                    defaultwtvalue=1
                except:
                    defaultwtvalue=0
                wtkeywordtovalue[weightkeyword]=defaultwtvalue

   
        proptovalue['global']=globaldic
        for wtkeyword,wtvalue in wtkeywordtovalue.items():
            proptovalue[wtkeyword]=wtvalue

        tptoproptovalue[tp]=proptovalue


    return tptoproptovalue


def WriteCSVFile(listoftpdics,nvtprops,molname):
    firsttpdic=listoftpdics[0]
    firsttpdicvalues=list(firsttpdic.values())
    firsttppointvalues=firsttpdicvalues[0]
    firstglobaldic=firsttppointvalues['global']
    nvtproperties=['Surf_Ten']
    with open('data'+'_'+molname+'.csv', mode='w') as write_file:
        writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for keyword,value in firstglobaldic.items():
            ls=['Global',keyword,value]
            writer.writerow(ls) 
        writer.writerow([])
        listofcommentrows=[]
        listoftprows=[]
        listoftpcommentrows=[]
        for tpdic in listoftpdics:
            for tp,proptovalue in tpdic.items():
                citation=proptovalue['cite']
                refls=['# Ref',citation]
                propls=['T','P']
                propvaluels=[str(tp[0]),str(tp[1])+' '+'atm']
                properrls=[]
                properrvaluels=[]
                commentpropvaluels=['# '+str(tp[0]),str(tp[1])+' '+'atm']

                ls=[refls,commentpropvaluels]
                for prop,value in proptovalue.items():
                    if prop!='cite' and 'wt' not in prop and 'err' not in prop and prop!='T' and prop!='P' and 'global' not in prop:
                        if value==None:
                            continue
                        if nvtprops==False and prop in nvtproperties:
                            continue

                        propls.append(prop)
                        propvaluels.append(value)
                        wtkeyword=prop+'_wt'
                        wtvalue=proptovalue[wtkeyword]
                        propls.append(wtkeyword)
                        propvaluels.append(wtvalue)
                        errkeyword=prop+'_err'
                        errvalue=proptovalue[errkeyword]
                        if errvalue==None:
                            errvalue=0
                        properrls.append(errkeyword)
                        properrvaluels.append(errvalue)
                properrls[0]='# '+str(properrls[0])
                properrvaluels[0]='# '+str(properrvaluels[0])

                ls.append(properrls) 
                ls.append(properrvaluels)
                listofcommentrows.append(ls)
                listoftpcommentrows.append(propls)
                listoftprows.append(propvaluels)
        for grpls in listofcommentrows:
            for ls in grpls:
                writer.writerow(ls)
            writer.writerow([])
        first=listoftpcommentrows[0]
        writer.writerow(first)
        for ls in listoftprows:
            writer.writerow(ls)
                        

def GenerateLiquidCSVFile(nvtprops,listoftptopropdics,molnamelist):
    for i in range(len(listoftptopropdics)):
        tptoproptovalue=listoftptopropdics[i]
        molname=molnamelist[i]
        tptoproptovalue=AddDefaultValues(tptoproptovalue)
        WriteCSVFile([tptoproptovalue],nvtprops,molname)
    

def ReadInPoltypeFiles(poltypepathlist):
    dimertinkerxyzfileslist=[]
    dimerenergieslist=[]
    dimersplogfileslist=[]
    keyfilelist=[]
    xyzfilelist=[]
    molnamelist=[]
    molfilelist=[]

    for poltypepath in poltypepathlist:
        dimertinkerxyzfiles=[]
        dimerenergies=[]
        dimersplogfiles=[]
        curdir=os.getcwd()
        os.chdir(poltypepath)
        keyfile=os.path.join(poltypepath,'final.key')
        xyzfile=os.path.join(poltypepath,'final.xyz')
        if not os.path.exists(keyfile) or not os.path.exists(xyzfile):
            raise ValueError('Final xyz or keyfile does not exist in '+str(poltypepath))
        if os.path.isdir('vdw'):
            os.chdir('vdw')
            files=os.listdir()
            spext='_sp.log'
            for f in files:
                if spext in f:
                    tempxyzfile=f.replace(spext,'.xyz')
                    dimertinkerxyzfiles.append(os.path.join(os.getcwd(),tempxyzfile))
                    dimersplogfiles.append(os.path.join(os.getcwd(),f))
            os.chdir('..')
        else:
            raise ValueError('vdw folder does not exist for '+poltypepath)

        files=os.listdir()
        for f in files:
            if '.' in f:
                filesplit=f.split('.')
                ext=filesplit[1]
                if ext=='sdf': # assume poltype input sdf file
                    molname=filesplit[0]
                    molfile=f


        dimerenergies=ReadQMLogFile(dimersplogfiles)
        molfile=os.path.join(poltypepath,molfile)
        os.chdir(curdir)
        dimertinkerxyzfileslist.append(dimertinkerxyzfiles)
        dimerenergieslist.append(dimerenergies)
        dimersplogfileslist.append(dimersplogfiles)
        keyfilelist.append(keyfile)
        xyzfilelist.append(xyzfile)
        molnamelist.append(molname)
        molfilelist.append(molfile)

    return keyfilelist,xyzfilelist,dimertinkerxyzfileslist,dimerenergieslist,molnamelist,molfilelist,dimersplogfileslist
 

def GrabMonomerEnergy(line,Hartree2kcal_mol):
    linesplit=line.split()
    energy=float(linesplit[4]) 
    monomerenergy=(energy)*Hartree2kcal_mol
    return monomerenergy

def CheckIfLogFileUsingGaussian(f):
    use_gaus=False
    temp=open(f,'r')
    results=temp.readlines()
    temp.close()
    for line in results:
        if 'Entering Gaussian System' in line:
            use_gaus=True
            break 
    return use_gaus

 
def ReadQMLogFile(dimersplogfiles):
    Hartree2kcal_mol=627.5095
    dimerenergies=[]
    for f in dimersplogfiles:
        use_gaus=CheckIfLogFileUsingGaussian(f)
        tmpfh=open(f,'r')
        if use_gaus==True:
            frag1calc=False
            frag2calc=False
            for line in tmpfh:
                if 'Counterpoise: corrected energy =' in line:
                    dimerenergy=float(line.split()[4])*Hartree2kcal_mol
                elif 'Counterpoise: doing DCBS calculation for fragment   1' in line:
                    frag1calc=True
                    frag2calc=False
                elif 'Counterpoise: doing DCBS calculation for fragment   2' in line:
                    frag1calc=False
                    frag2calc=True
                elif 'SCF Done' in line:
                    if frag1calc:
                        frag1energy=GrabMonomerEnergy(line,Hartree2kcal_mol)
                    elif frag2calc:
                        frag2energy=GrabMonomerEnergy(line,Hartree2kcal_mol)
            interenergy=dimerenergy-(frag1energy+frag2energy)
        else:
            for line in tmpfh:
                if 'CP Energy =' in line and 'print' not in line:
                    linesplit=line.split()
                    interenergy=float(linesplit[3])*Hartree2kcal_mol
        dimerenergies.append(interenergy)     
        tmpfh.close()
    

    return dimerenergies


def GrabVdwTypeLinesFromFinalKey(keyfilelist,vdwtypeslist):
    vdwtypelineslist=[]
    for i in range(len(keyfilelist)):
        vdwtypelines=[]
        keyfile=keyfilelist[i]
        vdwtypes=vdwtypeslist[i]
        temp=open(keyfile,'r')
        results=temp.readlines()
        temp.close()
        for line in results:
            if 'vdw' in line:
                for vdwtype in vdwtypes:
                    if str(vdwtype) in line:
                        vdwtypelines.append(line)
        vdwtypelineslist.append(vdwtypelines)

    return vdwtypelineslist

def GenerateForceFieldFiles(vdwtypelineslist,moleculeprmfilename,fittypestogether):

    try:
        length=len(fittypestogether)
        array=np.array(fittypestogether)
        array=np.transpose(array)
                
    except:
        array=[]

    if not os.path.isdir('forcefield'):
        os.mkdir('forcefield')
    os.chdir('forcefield')
    vdwtypetoline={}
    vdwtypetored={}
    typestonothaveprmkeyword=[]
    if len(array)!=0:
        for row in array:
            firsttype=row[0] # this one we reference and dont append
            rest=row[1:]
            for vdwtype in rest:
                typestonothaveprmkeyword.append(int(vdwtype))
    for vdwtypelines in vdwtypelineslist:
        for line in vdwtypelines:
            linesplit=line.split()
            vdwtype=int(linesplit[1])
            last=float(linesplit[-1])
            linelen=len(linesplit)
            linesplit.append('#')
            if vdwtype not in typestonothaveprmkeyword:
                linesplit.append('PRM')
                linesplit.append('2')
                linesplit.append('3')
                if linelen==5 and last!=1:
                    linesplit.append('4')
            if linelen==5 and last!=1:
                vdwtypetored[vdwtype]=True
            else:
                vdwtypetored[vdwtype]=False

            newline=' '.join(linesplit)+'\n'
            vdwtypetoline[vdwtype]=newline    
    if len(array)!=0:
        for row in array:
            firsttype=row[0] # this one we reference and dont append
            rest=row[1:]
            for vdwtype in rest:
                line=vdwtypetoline[int(vdwtype)]
                red=vdwtypetored[int(vdwtype)]
                line=line.replace('\n','')
                line+=" EVAL 2 PRM['VDWS/%s']"%(str(firsttype)) 
                line+=" 3 PRM['VDWT/%s']"%(str(firsttype)) 
                if red==True:
                    line+=" 4 PRM['VDWD/%s']"%(str(firsttype)) 

                line+='\n'
                vdwtypetoline[int(vdwtype)]=line


    temp=open(moleculeprmfilename,'w')
    for vdwtype,line in vdwtypetoline.items():
        temp.write(line)
    temp.close()
    os.chdir('..')
    
def RemoveKeyWord(keypath,keystring):
    read=open(keypath,'r')
    results=read.readlines()
    read.close()
    tempname=keypath.replace('.key','-t.key')
    temp=open(tempname,'w')
    for line in results:
        if keystring not in line:
            temp.write(line)
    temp.close()
    os.remove(keypath)
    os.rename(tempname,keypath)  


def CommentOutVdwLines(keypath,vdwtypes):
    read=open(keypath,'r')
    results=read.readlines()
    read.close()
    tempname=keypath.replace('.key','-t.key')
    temp=open(tempname,'w')
    for lineidx in range(len(results)):
        line=results[lineidx]
        for vdwtype in vdwtypes:
            if str(vdwtype) in line and 'vdw' in line:
                newline='# '+line
                results[lineidx]=newline
    for line in results:
        temp.write(line) 
    


    temp.close()
    os.remove(keypath)
    os.rename(tempname,keypath)  


def GenerateLiquidTargetsFolder(gaskeyfilelist,gasxyzfilelist,liquidkeyfilelist,liquidxyzfilelist,datacsvpathlist,densitylist,originalliquidfolder,prmfilepath,moleculeprmfilename,vdwtypeslist,addwaterprms,molnamelist):
    liquidfolderlist=[]
    for i in range(len(gaskeyfilelist)):
        gaskeyfile=gaskeyfilelist[i]
        gasxyzfile=gasxyzfilelist[i]
        liquidkeyfile=liquidkeyfilelist[i]
        liquidxyzfile=liquidxyzfilelist[i]
        datacsvpath=datacsvpathlist[i]
        vdwtypes=vdwtypeslist[i]
        molname=molnamelist[i]
        if not os.path.isdir('targets'):
            os.mkdir('targets')
        os.chdir('targets')
        liquidfolder=originalliquidfolder+'_'+molname
        liquidfolderlist.append(liquidfolder) 
        if not os.path.isdir(liquidfolder):
            os.mkdir(liquidfolder)
        os.chdir(liquidfolder)
        shutil.copy(gaskeyfile,os.path.join(os.getcwd(),'gas.key'))
        RemoveKeyWord(os.path.join(os.getcwd(),'gas.key'),'parameters')
        temp=open(os.path.join(os.getcwd(),'gas.key'),'a')
        if addwaterprms==True:
            waterlines=WaterParameters()
            for line in waterlines:
                temp.write(line+'\n')
        temp.close() 
        string='parameters '+moleculeprmfilename+'\n'
        AddKeyWord(os.path.join(os.getcwd(),'gas.key'),string)
        CommentOutVdwLines(os.path.join(os.getcwd(),'gas.key'),vdwtypes)
        shutil.copy(gasxyzfile,os.path.join(os.getcwd(),'gas.xyz'))
        shutil.copy(liquidkeyfile,os.path.join(os.getcwd(),'liquid.key'))
        CommentOutVdwLines(os.path.join(os.getcwd(),'liquid.key'),vdwtypes)
        shutil.copy(liquidxyzfile,os.path.join(os.getcwd(),'liquid.xyz'))
        os.remove(liquidxyzfile)
        if datacsvpath!=None:
            shutil.copy(datacsvpath,os.path.join(os.getcwd(),'data.csv'))
            os.remove(datacsvpath)
        os.chdir('..')
        os.chdir('..')

    return liquidfolderlist

def FindDimensionsOfMoleculeTinker(structurefilepath):
    veclist=[]
    temp=open(structurefilepath,'r')
    results=temp.readlines()
    temp.close()
    for line in results:
        linesplit=line.split()
        if len(linesplit)!=1 and '90.000000' not in line: # not line containing number of atoms
            vec=np.array([float(linesplit[2]),float(linesplit[3]),float(linesplit[4])])
            veclist.append(vec)

    pairs=list(itertools.combinations(veclist, 2))
    distlist=[]
    for pairidx in range(len(pairs)):
        pair=pairs[pairidx]
        progress=(pairidx*100)/len(pairs)
        dist=np.linalg.norm(np.array(pair[0])-np.array(pair[1]))
        distlist.append(dist)
    mindist=np.amax(np.array(distlist))
    return mindist

def ComputeBoxLength(xyzfile):
    vdwcutoff=12 #in angstrom
    longestdim=FindDimensionsOfMoleculeTinker(xyzfile)
    aaxis = 2*float(vdwcutoff)+longestdim+4
    return aaxis

def CreateSolventBox(axis,molnumber,prmfilepath,xyzeditpath,tinkerxyzname,molname):
    head,tail=os.path.split(tinkerxyzname)
    fullkey=tinkerxyzname.replace('.xyz','.key')
    key=tail.replace('.xyz','.key')
    shutil.copy(tinkerxyzname,os.path.join(os.getcwd(),tail))
    shutil.copy(fullkey,os.path.join(os.getcwd(),key))
    temp=open('xyzedit.in','w')
    temp.write(tail+'\n')
    temp.write('21'+'\n')
    temp.write(str(molnumber)+'\n')
    temp.write(str(axis)+','+str(axis)+','+str(axis)+'\n')
    temp.write('Y'+'\n')
    temp.close()
    cmdstr=xyzeditpath+' '+'<'+' '+'xyzedit.in'
    call_subsystem(cmdstr,wait=True)    
    os.replace(tail+'_2',molname+'_liquid.xyz') 
    liquidxyzfile=os.path.join(os.getcwd(),molname+'_liquid.xyz')
    os.remove(key)
    os.remove(tail)
    return liquidxyzfile

def call_subsystem(cmdstr,wait=False):
    p = subprocess.Popen(cmdstr, shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if wait==True:
        p.wait()
        if p.returncode != 0:
            raise ValueError("ERROR: " + cmdstr+' '+'path'+' = '+os.getcwd())


def InsertKeyfileHeader(keyfilename,moleculeprmfilename,axis):
    ewaldcutoff=7
    integrator="RESPA"
    thermostat="BUSSI"
    vdwcutoff=12
    polareps=.00001
    barostatmethod='montecarlo'
    string='parameters '+moleculeprmfilename+'\n'
    AddKeyWord(keyfilename,string)
    string='a-axis'+' '+str(axis)+'\n'
    AddKeyWord(keyfilename,string)
    string='archive'+'\n'
    AddKeyWord(keyfilename,string)
    string='integrator '+integrator+'\n'
    AddKeyWord(keyfilename,string)
    string='thermostat '+thermostat+'\n'
    AddKeyWord(keyfilename,string)
    string='ewald'+'\n'
    AddKeyWord(keyfilename,string)
    string='vdw-cutoff '+str(vdwcutoff)+'\n'
    AddKeyWord(keyfilename,string)
    string='ewald-cutoff '+str(ewaldcutoff)+'\n'
    AddKeyWord(keyfilename,string)
    string='polar-eps '+str(polareps)+'\n'
    AddKeyWord(keyfilename,string)
    string='polar-predict'+'\n'
    AddKeyWord(keyfilename,string)
    string='barostat'+' '+barostatmethod+'\n'
    AddKeyWord(keyfilename,string)
    string='vdw-correction'+'\n'
    AddKeyWord(keyfilename,string)


def AddKeyWord(keypath,string):
    read=open(keypath,'r')
    results=read.readlines()
    read.close()
    tempkeyname=keypath.replace('.key','-t.key')
    temp=open(tempkeyname,'w')
    temp.write(string)
    for line in results:
        temp.write(line)
    temp.close()
    os.remove(keypath)
    os.rename(tempkeyname,keypath)


def WaterParameters():
    lines=[]
    lines.append('atom        349   90    O     "AMOEBAWaterO"               8    15.999    2')
    lines.append('atom        350   91    H     "AMOEBAWaterH"               1     1.008    1')
    lines.append('vdw          90               3.4050     0.1100')
    lines.append('vdw          91               2.6550     0.0135      0.910')
    lines.append('bond         90   91          556.85     0.9572')
    lines.append('angle        91   90   91      48.70     108.50')
    lines.append('ureybrad     91   90   91      -7.60     1.5537')
    lines.append('multipole   349 -350 -350              -0.51966')
    lines.append('                                        0.00000    0.00000    0.14279')
    lines.append('                                        0.37928')
    lines.append('                                        0.00000   -0.41809')
    lines.append('                                       0.00000    0.00000    0.03881')
    lines.append('multipole   350  349  350               0.25983')
    lines.append('                                       -0.03859    0.00000   -0.05818')
    lines.append('                                       -0.03673')
    lines.append('                                        0.00000   -0.10739')
    lines.append('                                       -0.00203    0.00000    0.14412')
    lines.append('polarize    349          0.8370     0.3900    350')
    lines.append('polarize    350          0.4960     0.3900    349')
    return lines

def GenerateNewKeyFile(keyfile,prmfilepath,moleculeprmfilename,axis,addwaterprms,molname):
    liquidkeyfile=shutil.copy(keyfile,os.path.join(os.getcwd(),molname+'_liquid.key'))
    RemoveKeyWord(liquidkeyfile,'parameters')
    temp=open(prmfilepath,'r')
    results=temp.readlines()
    temp.close()
    temp=open(liquidkeyfile,'a')
    if addwaterprms==True:
        waterlines=WaterParameters()
        for line in waterlines:
            temp.write(line+'\n')
    temp.close() 
    
    InsertKeyfileHeader(liquidkeyfile,moleculeprmfilename,axis)

    return liquidkeyfile

def GenerateTargetFiles(keyfilelist,xyzfilelist,densitylist,rdkitmollist,prmfilepath,xyzeditpath,moleculeprmfilename,addwaterprms,molnamelist):
    gaskeyfilelist=[]
    gasxyzfilelist=[]
    liquidkeyfilelist=[]
    liquidxyzfilelist=[] 
    datacsvpathlist=[] 
    for i in range(len(rdkitmollist)):
        rdkitmol=rdkitmollist[i]
        xyzfile=xyzfilelist[i]
        keyfile=keyfilelist[i]
        density=densitylist[i]
        molname=molnamelist[i]
        mass=Descriptors.ExactMolWt(rdkitmol)*1.66054*10**(-27) # convert daltons to Kg
        axis=ComputeBoxLength(xyzfile)
        boxlength=axis*10**-10 # convert angstroms to m
        numbermolecules=int(density*boxlength**3/mass)
        liquidxyzfile=CreateSolventBox(axis,numbermolecules,prmfilepath,xyzeditpath,xyzfile,molname)
        gaskeyfile=keyfile.replace('final',molname+'Gas')
        gasxyzfile=xyzfile.replace('final',molname+'Gas')
        shutil.copy(keyfile,gaskeyfile)
        shutil.copy(xyzfile,gasxyzfile)
        
        name='data'+'_'+molname+'.csv'
        if os.path.exists(name):
            datacsvpath=os.path.join(os.getcwd(),name)
        else:
            datacsvpath=None
        liquidkeyfile=GenerateNewKeyFile(keyfile,prmfilepath,moleculeprmfilename,axis,addwaterprms,molname)
        gaskeyfilelist.append(gaskeyfile)
        gasxyzfilelist.append(gasxyzfile)
        liquidkeyfilelist.append(liquidkeyfile)
        liquidxyzfilelist.append(liquidxyzfile)
        datacsvpathlist.append(datacsvpath)
   

    return gaskeyfilelist,gasxyzfilelist,liquidkeyfilelist,liquidxyzfilelist,datacsvpathlist

def GenerateQMTargetsFolder(dimertinkerxyzfileslist,dimerenergieslist,liquidkeyfilelist,originalqmfolder,vdwtypeslist,molnamelist):
    qmfolderlist=[]
    for j in range(len(dimertinkerxyzfileslist)):
        dimertinkerxyzfiles=dimertinkerxyzfileslist[j]
        dimerenergies=dimerenergieslist[j]
        liquidkeyfile=liquidkeyfilelist[j]
        molname=molnamelist[j]
        vdwtypes=vdwtypeslist[j]
        os.chdir('targets')
        qmfolder=originalqmfolder+'_'+molname
        qmfolderlist.append(qmfolder)
        if not os.path.isdir(qmfolder):
            os.mkdir(qmfolder)
        os.chdir(qmfolder)
        newnamearray=[]
        newenergyarray=[]
        arr=np.arange(0,len(dimertinkerxyzfiles))
        arcwriter=open('all.arc','w')
        for i in range(len(arr)):
            value=arr[i]
            tinkerxyzfile=dimertinkerxyzfiles[i]
            dimerenergy=dimerenergies[i]
            head,tail=os.path.split(tinkerxyzfile)
            tinkerxyzfileprefix=tail.split('.')[0]
            newname=tinkerxyzfileprefix+str(value)
            temp=open(tinkerxyzfile,'r')
            results=temp.readlines()
            temp.close() 
            firstline=results[0]
            firstlinesplit=firstline.split()
            firstlinesplit.append(newname)
            newfirstline=' '.join(firstlinesplit)+'\n'
            results[0]=newfirstline
            newnamearray.append(newname)
            newenergyarray.append(dimerenergy) 
            for line in results:
                arcwriter.write(line)
        
        arcwriter.close()
        energywriter=open('qdata.txt','w')
        for i in range(len(arr)):
            value=arr[i]
            energy=newenergyarray[i]
            firstline='LABEL'+' '+str(value)+'\n'
            secondline='INTERACTION'+' '+str(energy)+'\n'
            energywriter.write(firstline)
            energywriter.write(secondline)
            energywriter.write('\n')
        energywriter.close()
        shutil.copy(liquidkeyfile,os.path.join(os.getcwd(),'liquid.key'))
        CommentOutVdwLines(os.path.join(os.getcwd(),'liquid.key'),vdwtypes)
        os.remove(liquidkeyfile)
        os.chdir('..')
        os.chdir('..')
 
    return qmfolderlist

def ReadLigandOBMol(structfname):
    tmpconv = openbabel.OBConversion()
    inFormat = openbabel.OBConversion.FormatFromExt(structfname)
    tmpconv.SetInFormat(inFormat)
    tmpmol = openbabel.OBMol()
    tmpconv.ReadFile(tmpmol, structfname)
    return tmpmol


def GenerateRdkitMol(ligandfilename):
    tmpmol=ReadLigandOBMol(ligandfilename)
    tmpconv = openbabel.OBConversion()
    tmpconv.SetOutFormat('mol')
    temp=ligandfilename.replace('.sdf','.mol')
    tmpconv.WriteFile(tmpmol, temp)
    rdkitmol=rdmolfiles.MolFromMolFile(temp,removeHs=False)
    return rdkitmol


def GenerateRdkitMolList(ligandfilenamelist):
    rdkitmollist=[]
    for ligandfilename in ligandfilenamelist:
        rdkitmol=GenerateRdkitMol(ligandfilename)
        rdkitmollist.append(rdkitmol)
    return rdkitmollist




def SanitizeMMExecutable(executable,tinkerdir):
    # Try to find Tinker executable with/without suffix
    if which(executable)!=None:
        return executable
    if tinkerdir is None:
        tinkerdir = os.getenv("TINKERDIR", default="")
    exe = os.path.join(tinkerdir, executable)
    if which(exe) is None:
        exe = exe[:-2] if exe.endswith('.x') else exe + '.x'
        if which(exe) is None:
            print("ERROR: Cannot find Tinker {} executable".format(executable))
            sys.exit(2)
    return exe

def which(program):
    def is_exe(fpath):
        try:
             return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
        except:
             return None
    
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    
    return None



def GenerateForceBalanceInputFile(moleculeprmfilename,qmfolderlist,liquidfolderlist,optimizefilepath,atomnumlist,liquid_equ_steps,liquid_prod_steps,liquid_timestep,liquid_interval,gas_equ_steps,gas_timestep,gas_interval,md_threads):
    head,tail=os.path.split(optimizefilepath)
    newoptimizefilepath=os.path.join(os.getcwd(),tail)
    shutil.copy(optimizefilepath,newoptimizefilepath)
    temp=open(newoptimizefilepath,'r')
    results=temp.readlines()
    temp.close()
    for lineidx in range(len(results)):
        line=results[lineidx]
        linesplit=line.split()
        if 'forcefield' in line:
            if linesplit[0]=='forcefield':
                newline='forcefield '+moleculeprmfilename+'\n'
                results[lineidx]=newline
    
           
    temp=open(newoptimizefilepath,'a')
    for i in range(len(qmfolderlist)):
        qmfolder=qmfolderlist[i]
        liquidfolder=liquidfolderlist[i] 
        atomnum=atomnumlist[i]
        results.append('$target'+'\n')
        results.append('name '+qmfolder+'\n')
        results.append('type Interaction_TINKER'+'\n')
        results.append('weight .5'+'\n')
        results.append('energy_denom 1.0'+'\n')
        results.append('energy_upper 20.0'+'\n')
        results.append('attenuate'+'\n')
        lastindex=str(atomnum)
        newindex=str(atomnum+1)
        lastnewindex=str(atomnum+1+2)
        results.append('fragment1 '+'1'+'-'+lastindex+'\n')
        results.append('fragment2 '+str(newindex)+'-'+lastnewindex+'\n')
        results.append('$end'+'\n')
        results.append('$target'+'\n')
        results.append('name '+liquidfolder+'\n')
        results.append('type Liquid_TINKER'+'\n')
        results.append('weight 1.0'+'\n')
        results.append('w_rho 1.0'+'\n')
        results.append('w_hvap 1.0'+'\n')
        results.append('liquid_equ_steps '+str(liquid_equ_steps)+'\n')
        results.append('liquid_prod_steps '+str(liquid_prod_steps)+'\n')
        results.append('liquid_timestep '+str(liquid_timestep)+'\n')
        results.append('liquid_interval '+str(liquid_interval)+'\n')
        results.append('gas_equ_steps '+str(gas_equ_steps)+'\n')
        results.append('gas_prod_steps '+str(gas_prod_steps)+'\n')
        results.append('gas_timestep '+str(gas_timestep)+'\n')
        results.append('gas_interval '+str(gas_interval)+'\n')
        results.append('md_threads '+str(md_threads)+'\n')
        results.append('$end'+'\n')
    for line in results:
        temp.write(line)
    temp.close()


def GrabMaxTypeNumber(parameterfile):
    maxnumberfromprm=1
    temp=open(parameterfile,'r')
    results=temp.readlines()
    temp.close()
    for line in results:
        if 'atom' in line:
            linesplit=line.split()
            atomtype=int(linesplit[1])
            if atomtype>maxnumberfromprm:
                maxnumberfromprm=atomtype
    return maxnumberfromprm

def GrabMinTypeNumber(parameterfile):
    minnumberfromprm=10000
    temp=open(parameterfile,'r')
    results=temp.readlines()
    temp.close()
    for line in results:
        if 'atom' in line:
            linesplit=line.split()
            atomtype=int(linesplit[1])
            if atomtype<minnumberfromprm:
                minnumberfromprm=atomtype
    return minnumberfromprm


def GenerateTypeMaps(keyfilelist):
    oldtypetonewtypelist=[]
    currentmin=100000
    currentmax=0
    firsttime=False
    for keyfilename in keyfilelist:
        maxnumberfromkey=GrabMaxTypeNumber(keyfilename)
        minnumberfromkey=GrabMinTypeNumber(keyfilename)
        if firsttime==True:
            shift=np.abs(currentmax-maxnumberfromkey)+1+prevmaxnumberfromkey-prevminnumberfromkey
        else:
            shift=0
        firsttime=True
        if minnumberfromkey<currentmin:
            currentmin=minnumberfromkey
        if maxnumberfromkey>currentmax:
            currentmax=maxnumberfromkey
        types=np.arange(minnumberfromkey,maxnumberfromkey+1,1)
        shiftedtypes=types+shift
        oldtypetonewtype=dict(zip(types,shiftedtypes))
        oldtypetonewtypelist.append(oldtypetonewtype)
        prevmaxnumberfromkey=maxnumberfromkey
        prevminnumberfromkey=minnumberfromkey
    return oldtypetonewtypelist

def ShiftVdwTypesLines(vdwtypelineslist,oldtypetonewtypelist):
    newvdwtypelineslist=[]
    for i in range(len(vdwtypelineslist)):
        newvdwtypelines=[]
        vdwtypelines=vdwtypelineslist[i]
        for vdwtypeline in vdwtypelines:
            oldtypetonewtype=oldtypetonewtypelist[i]
            linesplit=vdwtypeline.split()
            vdwtype=int(linesplit[1])
            newvdwtype=str(oldtypetonewtype[vdwtype])
            linesplit[1]=newvdwtype
            newline=' '.join(linesplit) 
            newvdwtypelines.append(newline)
        newvdwtypelineslist.append(newvdwtypelines)

    return newvdwtypelineslist 

def ShiftTypes(typeslist,oldtypetonewtypelist):
    if typeslist==None:
        return typeslist
    newtypeslist=[]
    for i in range(len(typeslist)):
        types=typeslist[i]
        oldtypetonewtype=oldtypetonewtypelist[i]
        newtypes=[]
        for ttype in types:
            ttype=int(ttype)
            newtype=str(oldtypetonewtype[ttype])
            newtypes.append(newtype)
        newtypeslist.append(newtypes) 

    return newtypeslist

def FilterHighEnergy(dimertinkerxyzfileslist,dimerenergieslist,dimersplogfileslist):
    tol=15 # kcal/mol
    newdimertinkerxyzfileslistoflist=[]
    newdimerenergieslistoflist=[]
    for j in range(len(dimertinkerxyzfileslist)):
        newdimertinkerxyzfileslist=[]
        newdimerenergieslist=[]

        prefixtoxyzfilelist={}
        prefixtoenergylist={}
        prefixtooriginalenergylist={}
        xyzfilelist=dimertinkerxyzfileslist[j]
        energylist=dimerenergieslist[j]
        splogfilelist=dimersplogfileslist[j]
        for i in range(len(xyzfilelist)):
            xyzfile=xyzfilelist[i]
            energy=energylist[i]
            splogfile=splogfilelist[i]
         
            filesplit=splogfile.split('_')
            filesplit=filesplit[:-2]
            prefix=''.join(filesplit)
            if prefix not in prefixtoxyzfilelist.keys():
                prefixtoxyzfilelist[prefix]=[]
                prefixtoenergylist[prefix]=[]
            prefixtoxyzfilelist[prefix].append(xyzfile)
            prefixtoenergylist[prefix].append(energy)
        for prefix,energyarray in prefixtoenergylist.items():
            normenergyarray=[i-min(energyarray) for i in energyarray]
            prefixtoenergylist[prefix]=normenergyarray
            prefixtooriginalenergylist[prefix]=energyarray

        for prefix,energyarray in prefixtoenergylist.items():
            normenergyarray=prefixtoenergylist[prefix]
            energyarray=prefixtooriginalenergylist[prefix]

            xyzfilelist=prefixtoxyzfilelist[prefix]
            for eidx in range(len(normenergyarray)):
                e=normenergyarray[eidx]
                originale=energyarray[eidx]
                xyzfile=xyzfilelist[eidx]
                if e>=tol:
                    pass
                else:
                    newdimertinkerxyzfileslist.append(xyzfile)    
                    newdimerenergieslist.append(originale)
        newdimertinkerxyzfileslistoflist.append(newdimertinkerxyzfileslist)
        newdimerenergieslistoflist.append(newdimerenergieslist)


    return newdimertinkerxyzfileslistoflist,newdimerenergieslistoflist


if poltypepathlist!=None:
    keyfilelist,xyzfilelist,dimertinkerxyzfileslist,dimerenergieslist,molnamelist,molfilelist,dimersplogfileslist=ReadInPoltypeFiles(poltypepathlist)
    dimertinkerxyzfileslist,dimerenergieslist=FilterHighEnergy(dimertinkerxyzfileslist,dimerenergieslist,dimersplogfileslist)
    densitylist=[ls[0] for ls in density_list] # just grab first density 
    GenerateLiquidCSVFile(nvtprops,listoftptopropdics,molnamelist)
    prmfilepath=os.path.join(os.path.split(__file__)[0],'amoebabio18.prm')
    optimizefilepath=os.path.join(os.path.split(__file__)[0],'optimize.in')
    xyzeditpath='xyzedit'
    tinkerdir=None
    xyzeditpath=SanitizeMMExecutable(xyzeditpath,tinkerdir)
    vdwtypelineslist=GrabVdwTypeLinesFromFinalKey(keyfilelist,vdwtypeslist)
    oldtypetonewtypelist=GenerateTypeMaps(keyfilelist)
    vdwtypelineslist=ShiftVdwTypesLines(vdwtypelineslist,oldtypetonewtypelist)
    fittypestogether=ShiftTypes(fittypestogether,oldtypetonewtypelist)
    vdwtypeslist=ShiftTypes(vdwtypeslist,oldtypetonewtypelist)
    moleculeprmfilename='molecule.prm'
    GenerateForceFieldFiles(vdwtypelineslist,moleculeprmfilename,fittypestogether)
    rdkitmollist=GenerateRdkitMolList(molfilelist)
    atomnumlist=[rdkitmol.GetNumAtoms() for rdkitmol in rdkitmollist]
    gaskeyfilelist,gasxyzfilelist,liquidkeyfilelist,liquidxyzfilelist,datacsvpathlist=GenerateTargetFiles(keyfilelist,xyzfilelist,densitylist,rdkitmollist,prmfilepath,xyzeditpath,moleculeprmfilename,addwaterprms,molnamelist)
    liquidfolder='Liquid'
    qmfolder='QM'
    
    liquidfolderlist=GenerateLiquidTargetsFolder(gaskeyfilelist,gasxyzfilelist,liquidkeyfilelist,liquidxyzfilelist,datacsvpathlist,densitylist,liquidfolder,prmfilepath,moleculeprmfilename,vdwtypeslist,addwaterprms,molnamelist)
    qmfolderlist=GenerateQMTargetsFolder(dimertinkerxyzfileslist,dimerenergieslist,liquidkeyfilelist,qmfolder,vdwtypeslist,molnamelist)    
    GenerateForceBalanceInputFile(moleculeprmfilename,qmfolderlist,liquidfolderlist,optimizefilepath,atomnumlist,liquid_equ_steps,liquid_prod_steps,liquid_timestep,liquid_interval,gas_equ_steps,gas_timestep,gas_interval,md_threads) 
