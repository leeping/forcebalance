import os, numpy as np
import pyilt2
import sys
import cirpy
import pint
from pint import UnitRegistry
import csv
import getopt
from tqdm import tqdm
import shutil
import openbabel
from rdkit.Chem import rdmolfiles
from rdkit.Chem import Descriptors
import itertools
from bs4 import BeautifulSoup # for parsing NIST
import requests
import re
import pandas as pd
import numpy as np
from rdkit import Chem
import subprocess


smiles=None
truename=None
printnames=False
searchILDB=False # search NIST by default
poltypepath=None
rep='names'
roomtemp=295.5
lowertol=10
uppertol=10
vdwtypes=None
liquid_equ_steps=10000
liquid_prod_steps=100000
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
skipNIST=False
density=1000 # default water density

opts, xargs = getopt.getopt(sys.argv[1:],'',["smiles=","truename=",'printnames','searchILDB','poltypepath=','vdwtypes=','liquid_equ_steps=','liquid_prod_steps=','liquid_timestep=','liquid_interval=','gas_equ_steps=','gas_prod_steps=','gas_timestep=','gas_interval=','md_threads=','liquid_prod_time=','gas_prod_time=','nvtpros=','skipNIST','density='])
for o, a in opts:
    if o in ("--smiles"):
        smiles=a
    elif o in ("--truename"):
        truename=a.capitalize() # the database has first letter capitalized 
    elif o in ("--printnames"):
        printnames=True
    elif o in ("--searchILDB"):
        searchILDB=False
    elif o in ("--poltypepath"):
        poltypepath=a
    elif o in ("--vdwtypes"):
        vdwtypes=a.split(',')
        vdwtypes=[i.lstrip().rstrip() for i in vdwtypes]
    elif o in ("--liquid_equ_steps"):
        liquid_equ_steps=int(a)
    elif o in ('--liquid_prod_steps'):
        liquid_prod_steps=int(a) 
    elif o in ('--liquid_timestep'):
        liquid_timestep=int(a)
    elif o in ('--liquid_interval'):
        liquid_interval=float(a)
    elif o in ('--gas_equ_steps'):
        gas_equ_steps=int(a)
    elif o in ('--gas_prod_steps'):
        gas_prod_steps=int(a)
    elif o in ('--gas_timestep'):
        gas_timestep=float(a)
    elif o in ('--gas_interval'):
        gas_interval=float(a)
    elif o in ('--md_threads'):
        md_threads=int(a)
    elif o in ('--liquid_prod_time'):
        liquid_prod_time=float(a)
    elif o in ('--gas_prod_time'):
        gas_prod_time=float(a)
    elif o in ('--nvtprops'):
        nvtprops=True
    elif o in ('--skipNIST'):
        skipNIST=True
    elif o in ('--density'):
        density=float(a)


gas_prod_steps=int(1000000*gas_prod_time/gas_timestep)
liquid_prod_steps=int(1000000*liquid_prod_time/liquid_timestep)

# originally taken from https://github.com/sabidhasan/NistPy
# modified to take inputs for force balance
class NistCompoundById(object):
	'''This class creates a NIST compound using a provided NIST code.
	NIST identifying codes consist of a letter followed by several numbers'''

	#A search result is NistResult object. It's instantiated is False. This is for user 
	#to see whether the object has been instantaited in a friendly manner. Really you can
	#just do type(object); if it's NistCpdById then it is instantiated; if NistResult, it's not.
	INSTANTIATED = True

	@staticmethod
	def extract_CAS(cas_string):
		'''Extracts a CAS number using RegEx from a given string cas_string'''
		reg = re.search(r'\d{2,7}-\d{2}-\d', cas_string)
		if reg:
			return reg.group(0)
		return None
			
	def __init__(self, id, souped_page=None):
		if souped_page==None:
			cpd_url = "http://webbook.nist.gov/cgi/cbook.cgi?ID=%s" % id
			cpd_soup = download_page(cpd_url)
		
			if cpd_soup is None:
				#download failed
				return None
		else:
			#we are given a souped page already!
			cpd_soup = souped_page	
		
		#check for erroneous number
		if "Registry Number Not Found" in cpd_soup:
			print("<Compound %s not found in registry>" % id)
			return None

		#Set compound name
		self.name = cpd_soup.find('h1').text
		self.id = id
		
		
				
	def __str__(self):
		return 'Compound %s' % self.name
		
	def __repr__(self):
		return "NistCompoundById('%s')" % self.id

	def _get_fluid_data(self,cid,temp,plow=0,phigh=2,pinc=.01,table=0):

                url='https://webbook.nist.gov/cgi/fluid.cgi?Action=Load&ID='+str(cid)+'&Type=IsoTherm&Digits=5&PLow='+str(plow)+'&PHigh='+str(phigh)+'&PInc='+str(pinc)+'&T='+str(temp)+'&RefState=DEF&TUnit=K&PUnit=atm&DUnit=kg%2Fm3&HUnit=kJ%2Fmol&WUnit=m%2Fs&VisUnit=uPa*s&STUnit=N%2Fm'
                prop_soup = download_page(url)
                prop_table = prop_soup.find_all("table")[table].find_all('tr')
                table=[]
                for row in prop_table:
                    subrow=[]
                    if len(row.find_all('td')) > 0:
                        for j in range(14):
                            subrow.append(row.find_all('td')[j].text)
                    elif len(row.find_all('th')) > 0:
                        for j in range(14):
                            subrow.append(row.find_all('th')[j].text)
                    table.append(subrow)

                return np.array(table)

	def _get_fluid_Saturation_data(self,cid,tlow=50,thigh=350,tinc=1,table=0): # for surface tension
                url='https://webbook.nist.gov/cgi/fluid.cgi?Action=Load&ID='+str(cid)+'&Type=SatP&Digits=5&THigh='+str(thigh)+'&'+str(tlow)+'=50&TInc='+str(tinc)+'&RefState=DEF&TUnit=K&PUnit=atm&DUnit=kg%2Fm3&HUnit=kJ%2Fmol&WUnit=m%2Fs&VisUnit=uPa*s&STUnit=N%2Fm'
                prop_soup = download_page(url)
                prop_table = prop_soup.find_all("table")[table].find_all('tr')
                table=[]
                for row in prop_table:
                    subrow=[]
                    if len(row.find_all('td')) > 0:
                        for j in range(14):
                            subrow.append(row.find_all('td')[j].text)
                    elif len(row.find_all('th')) > 0:
                        for j in range(14):
                            subrow.append(row.find_all('th')[j].text)
                    table.append(subrow)


                return np.array(table)

	def _get_PhaseChange_data(self,cid,table=2):

                prop_url = "https://webbook.nist.gov/cgi/cbook.cgi?ID="+str(cid)+"&Mask=4#"+"Thermo-Phase"
                prop_soup = download_page(prop_url)

                tables=prop_soup.find_all("table")
                
                thetab=tables[2]
                content=thetab.find_all('td')
                hcontent=thetab.find_all('th')
                
                table=[]
                header=[]
                for row in hcontent:
                    header.append(row.text)
                foundenthalpy=False
                for e in header:
                    if 'vapH' in e:
                        foundenthalpy=True
                if foundenthalpy==False:
                    return table
                table.append(header)
                count=0
                ls=[]
                for rowidx in range(len(content)):
                    row=content[rowidx]
                    if count==len(header):
                        count=0
                    if count==0:
                        if len(ls)!=0:
                            table.append(ls)
                        ls=[]
                    value=row.text
                    count+=1
                    if value.strip()=='':
                        value='N/A' 
                    ls.append(value)
         

                return table

		
	

def download_page(url):
	'''This function downloads a page given and URL. If fail, it returns a status-code'''	
	page = requests.get(url)
	page_content = page.content
	#test for erroneous download
	if page.status_code != 200:
		print("<Error downloading web page. Error status code %s>" % cpd_page.status_code)
		return None
		
	return BeautifulSoup(page_content, 'html.parser')

class NistResult(object):
	'''NistResult class is used to return search results. This class is like a
	NistCompoundById but only has id and name. It has the instantiate method
	to elaborate and return NistCompoundById objects'''

	INSTANTIATED = False
	
	def __init__(self, id_num, name, souped_page=None):
		self.id = id_num
		self.name = name
		self.page = souped_page

	def instantiate(self):
		#This will instantiate the compound
		if self.page is None:
			self.page = download_page("http://webbook.nist.gov/cgi/cbook.cgi?ID=%s" % self.id)
			
		return NistCompoundById(self.id, souped_page=self.page)

	def __str__(self):
		return self.id

	def __repr__(self):
		return 'NistResult(%s, %s)' % (self.id, self.name)

def search_nist(options=None, instantiate=False, **kwargs):
	'''Search NIST database for string search_str using the search keyword [formula, name, inchi key, CAS number, 
	structure]. If instantiate is selected, then it returns NistCompoundById object (which has a bunch of data within it)
	Otherwise it returns a NistResult object, which only has the name and ID code and must be called with the
	.instantiate() method before further use'''

	#Error checking
	if len(kwargs) != 1:
		print("<Expected one keyword, obtained %s>" % len(kwargs))
		return None
	search_type=list(kwargs.keys())[0]
	search_str=list(kwargs.values())[0]
		
	#Check for value of kwarg	
	types = {
	'formula': "http://webbook.nist.gov/cgi/cbook.cgi?Formula=%s" % search_str,
	'name' : "http://webbook.nist.gov/cgi/cbook.cgi?Name=%s" % search_str,
	'inchi' : "http://webbook.nist.gov/cgi/cbook.cgi?InChI=%s" % search_str,
	'cas' : "http://webbook.nist.gov/cgi/cbook.cgi?ID=%s" % search_str,
	}
	if not(search_type) in types:
		print("<Got unexpected search type. Only these search types allowed %s>" % types.keys())
		return None
		
	#Download the search page, and check to see whether it's a single compound
	search_page = download_page(types[search_type])
	#will be returned to user
	search_results = []			
	
	if "Search Results" in search_page.text:
		#Loop through all results. The results are in an ordered list!
		
		for raw_result in search_page.ol.find_all('li'):
			id_code = re.search(r'ID=([A-Za-z]\d{1,})', raw_result.a['href']).group(1)
			name = raw_result.a.text
			if instantiate == True:
				search_results.append(NistResult(id_code, name).instantiate())
			else:
				search_results.append(NistResult(id_code, name))
	#there is only one search result!
	else:
		#Find the ID (basically it's the ID-like text that appears the most on the page!)
		links_on_page = []
		#loop through all the links
		for link in search_page.find_all('a'):
			curr_href = link.attrs.get('href')
			#sometimes, there is no hfref (eg. it's an anchor or something), so check for NoneType
			if curr_href is not None and re.search(r'ID=([A-Za-z]\d{1,})', curr_href):
				#add the current link's ID code
				#group(0) is the entire search result, so group(1) is the bracketed thing in the search query
				links_on_page.append(re.search(r'ID=([A-Za-z]\d{1,})', curr_href).group(1))
		if len(links_on_page)==0:
			raise ValueError('NIST Name search failed')
		id_code = max(links_on_page, key=links_on_page.count)
		
		name = search_page.title.text
		
		if instantiate == True:
			search_results.append(NistResult(id_code, name, souped_page=search_page).instantiate())
		else:
			search_results.append(NistResult(id_code, name, souped_page=search_page))

	return search_results	




def ConvertSmilestoNames(smiles,rep):
    results=cirpy.query(smiles, rep)
    if len(results)==0:
        raise ValueError('SMILES could not be converted to name')
    else:
        return results[0].value

def SanitizeUnit(currentunit):
    if currentunit!=None:
        currentunit=currentunit.replace('m3','m**3')

    return currentunit



def QueryLiquidProperties(names,truename=None,printnames=False):
    if truename==None:
        foundname=False
    else:
        foundname=True
        thename=truename
    foundnames=[]
    foundhit=False
    ureg = UnitRegistry()
    fbabbrtotargetunits={'Eps0':'debye**2/(nm**3 .J)','Eps0_err':'debye**2/(nm**3 .J)','Cp':'kJ/(mol.K)','Cp_err':'kJ/(mol.K)','Kappa':'1/bar','Kappa_err':'1/bar','Alpha':'1/K','Alpha_err':'1/K','Surf_Ten':'mJ/m**2','Surf_Ten_err':'mJ/m**2','T':'K','P':'atm','Rho':'kg/m**3','Rho_err':'kg/m**3','Hvap':'kJ/mol','Hvap_err':'kJ/mol'}
    nistnametofbabbr={'Enthalpy_of_vaporization_or_sublimation':'Hvap', 'Delta[Enthalpy_of_vaporization_or_sublimation]':'Hvap_err','Surface_tension_liquid-gas':'Surf_Ten', 'Delta[Surface_tension_liquid-gas]':'Surf_Ten_err','Relative_permittivity_at_zero_frequency':'Eps0', 'Delta[Relative_permittivity_at_zero_frequency]':'Eps0_err','Isothermal_compressibility':'Kappa', 'Delta[Isothermal_compressibility]':'Kappa_err','Isobaric_coefficient_of_volume_expansion':'Alpha', 'Delta[Isobaric_coefficient_of_volume_expansion]':'Alpha_err','Heat_capacity_at_constant_pressure*':'Cp', 'Delta[Heat_capacity_at_constant_pressure*]':'Cp_err','Pressure':'P', 'Temperature':'T','Specific_density':'Rho', 'Delta[Specific_density]':'Rho_err','Specific_density*':'Rho', 'Delta[Specific_density*]':'Rho_err','Heat_capacity_at_constant_pressure':'Cp', 'Delta[Heat_capacity_at_constant_pressure]':'Cp_err'}
    desiredprops=['Isothermal compressibility','Heat capacity at constant pressure','Density','Isobaric coefficient of volume expansion','Surface tension liquid-gas','Relative permittivity',"Enthalpy of vaporization or sublimation"]
    propertynamearray=list(fbabbrtotargetunits.keys())
    refstoproptoarray={}
    for name in names:
        if foundname==True and printnames==False:
            if name==thename:
                pass
            else:
                continue

        for prop,aprop in tqdm(pyilt2.prop2abr.items(),desc='Property search for '+name):
            if prop in desiredprops:
                results = pyilt2.query(comp = name,prop=aprop)

                length=len(results.refs)
                if length>0:
                    foundhit=True
                    if name not in foundnames:
                        foundnames.append(name)
                    if printnames==False: 
                        if foundname==False:
                            foundname=True
                            thename=name
                        for hit in results.refs:
                            hitrefdict=hit.refDict
                            hitref=hitrefdict['ref']
                            if hitref not in refstoproptoarray.keys():
                                ls=[None for i in propertynamearray]
                                dic=dict(zip(propertynamearray,ls))
                                refstoproptoarray[hitref]=dic
                            try:
                                dataset=hit.get()
                            except:
                                print('Error, in accessing database!')
                                continue
                            props=dataset.physProps
                            units=dataset.physUnits
                            data=dataset.data
                            fullref=dataset.fullcite
                            refstoproptoarray[hitref]['cite']=fullref
                            refstoproptoarray[hitref]['name']=thename
                            for pidx in range(len(props)):
                                p=props[pidx]
                                currentunit=units[pidx]
                                if p in nistnametofbabbr.keys() and currentunit==None:
                                    break
                                currentunit=SanitizeUnit(currentunit)
                                col=data[:,pidx]
                                if 'fraction' not in p and 'MolaLity' not in p and 'MolaRity' not in p and 'Mole' not in p and 'Molar' not in p:
                                    fbabbr=nistnametofbabbr[p]
                                    targetunit=fbabbrtotargetunits[fbabbr]
                                    if currentunit!=targetunit:
                                        value=str(1)
                                        string=value+' * '+currentunit+' to '+targetunit
                                        src, dst = string.split(' to ') 
                                        Q_ = ureg.Quantity
                                        convobj=Q_(src).to(dst)
                                        convfactor=convobj._magnitude
                                        converted=col*convfactor
                                        refstoproptoarray[hitref][fbabbr]=converted
                                    else:
                                        refstoproptoarray[hitref][fbabbr]=col
    if printnames==True:
        print('possible names',foundnames)
        sys.exit()
    if foundhit==False:
        raise ValueError('no references found for name '+thename)
    return refstoproptoarray

def SortReferences(refstoproptoarray,searchILDB):
    sortedrefstoproptoarray={}
    refstodates={}
    for ref in refstoproptoarray.keys():
        if searchILDB==True:
            first='('
            last=')'
            start = ref.rindex( first ) + len( first )
            end = ref.rindex( last, start )
            string=ref[start:end]
            numstring=[e for e in string if e.isdigit()==True]
            numstring=''.join(numstring)
            num=int(numstring)
        else:
            if ',' in ref:
                refsplit=ref.split(',')
                datestring=refsplit[-1]
                datestring=datestring.lstrip().rstrip()
                num=int(datestring)
            else:
                num=0
        refstodates[ref]=num
    sortedreftodates=dict(sorted(refstodates.items(), key=lambda item: item[1],reverse=True)) # later dates first
    for sortedref,date in sortedreftodates.items():
        proptoarray=refstoproptoarray[sortedref]
        sortedrefstoproptoarray[sortedref]=proptoarray
    

    return sortedrefstoproptoarray

def PropertyToReferences(sortedrefstoproptoarray):
    proptorefs={}
    for sortedref,proptoarray in sortedrefstoproptoarray.items():
        for prop in proptoarray.keys():
            if prop!='cite' and prop!='name' and 'err' not in prop and prop!='T' and prop!='P':
                try:
                   value=proptoarray[prop]
                   if len(value)>0:
                       if prop not in proptorefs.keys():
                           proptorefs[prop]=[]
                       if sortedref not in proptorefs[prop]:
                           proptorefs[prop].append(sortedref)
                except:
                    continue
    if len(proptorefs.keys())==0:
        raise ValueError('Missing property data!')
    return proptorefs


def GrabUniqueProps(proptoarray):
    uniqueprops=[]
    for prop,array in proptoarray.items():
        if prop!='name' and prop!='cite' and 'err' not in prop and prop!='T' and prop!='P':
            try:
                length=len(array)
                if length>0:
                    uniqueprops.append(prop) 
            except:
                pass

    return tuple(set(uniqueprops))

def RoundTPData(sortedrefstoproptoarray,decimalplaces):
    for ref,proptoarray in sortedrefstoproptoarray.items():
        tarray=proptoarray['T']
        parray=proptoarray['P']
        try: # might not be an P data
            parray=[round(i,decimalplaces) for i in parray]
        except:
            pass

        first=tarray[0]
        if '-' in first:
            firstsplit=first.split('-')
            firsttemp=firstsplit[0].lstrip().rstrip()
            if firsttemp[-1]=='.':
                firsttemp=firsttemp[:-1]
            secondtemp=firstsplit[1].lstrip().rstrip()
            if secondtemp[-1]=='.':
                secondtemp=secondtemp[:-1]
            firsttemp=int(firsttemp)
            secondtemp=int(secondtemp)
            tarray=np.arange(firsttemp, secondtemp, 0.01).tolist()
        tarray=[float(i) for i in tarray]
        tarray=[round(i,decimalplaces) for i in tarray]
        sortedrefstoproptoarray[ref]['T']=tarray
        sortedrefstoproptoarray[ref]['P']=parray


    return sortedrefstoproptoarray

def CountTPFrequency(sortedrefstoproptoarray):
    uniquepropstotemppressurecounts={}
    for ref,proptoarray in sortedrefstoproptoarray.items():
        tarray=proptoarray['T']
        parray=proptoarray['P']
        setuniqueprops=GrabUniqueProps(proptoarray)
        if setuniqueprops not in uniquepropstotemppressurecounts.keys():
            uniquepropstotemppressurecounts[setuniqueprops]={}
        for i in range(len(tarray)):
            T=tarray[i]
            try:
                P=parray[i]
            except:
                P=1 # assume atmospheric but better be safe and check!
            tp=tuple([T,P])
            if tp not in uniquepropstotemppressurecounts[setuniqueprops].keys():
                uniquepropstotemppressurecounts[setuniqueprops][tp]=0
            uniquepropstotemppressurecounts[setuniqueprops][tp]+=1
    temppressurecounts={}
    for uniqueprop,tptocounts in uniquepropstotemppressurecounts.items():
        for tp in tptocounts.keys():
            if tp not in temppressurecounts.keys():
                temppressurecounts[tp]=0
            temppressurecounts[tp]+=1
    sortedtemppressurecounts=dict(sorted(temppressurecounts.items(), key=lambda item: item[1],reverse=True))
    
    return sortedtemppressurecounts 


def GrabNearestTPPoint(temp,tppoints):
    difftotppoint={}
    for tp in tppoints:
        t=tp[0]
        p=tp[1]
        diff=np.abs(t-temp)
        difftotppoint[diff]=tp
    mindiff=min(difftotppoint.keys())
    mintp=difftotppoint[mindiff]
    return mintp
               


def GrabTPPoints(proptoarray):
    tppoints=[]
    tarray=proptoarray['T']
    parray=proptoarray['P']
    for i in range(len(tarray)):
        T=tarray[i]
        try:
            P=parray[i]
        except:
            P=1
        tp=tuple([T,P])
        tppoints.append(tp)


    return tppoints


def GrabPropValue(T,tarray,array):
    try: # sometimes no error array:
        length=len(array)

    except: # return None if no error is reported
        return None
    for i in range(len(tarray)):
        t=tarray[i]
        value=array[i]
        if t==T:
            return value


def FindMinimalTPPointsAllProps(tppoint,proptorefs,sortedrefstoproptoarray,truename):
    tptoproptovalue={}
    T=tppoint[0]
    for prop,refs in proptorefs.items():
        for ref in refs:
            proptoarray=sortedrefstoproptoarray[ref]
            tppoints=GrabTPPoints(proptoarray)
            if 'cite' in proptoarray.keys():
                cite=proptoarray['cite']
            else:
                cite=ref
            if 'name' in proptoarray.keys():
                name=proptoarray['name']
            else:
                name=truename
            if tppoint in tppoints: # else need to find nearest TP point
                truepoint=tppoint 
            else:    
                othertppoint=GrabNearestTPPoint(T,tppoints)
                truepoint=othertppoint
            if truepoint not in tptoproptovalue.keys():
                tptoproptovalue[truepoint]={} 
                tptoproptovalue[truepoint]['cite']=cite
                tptoproptovalue[truepoint]['name']=name
            for prop,array in proptoarray.items():
                errprop=prop+'_err'
                length=0 
                if prop!='T' and prop!='P':
                    try:
                        length=len(array)
                        if length>0:
                            array=proptoarray[prop]
                            tarray=proptoarray['T']
                            value=GrabPropValue(T,tarray,array)
                            array=proptoarray[errprop]
                            errvalue=GrabPropValue(T,tarray,array)
                            tptoproptovalue[truepoint][prop]=value            
                            tptoproptovalue[truepoint][errprop]=errvalue   
                    except:


                        if prop not in tptoproptovalue[truepoint].keys():
                            tptoproptovalue[truepoint][prop]=None       
                            tptoproptovalue[truepoint][errprop]=None   

   
            break # find the first reference with data point, refs sorted by year, grab most recent 
  

    return tptoproptovalue


def GrabTPPointOutsideBound(tppoint,tol,tppoints,string):
    T=tppoint[0]
    if string=='low':
        bound=T-tol
    elif string=='upper':
        bound=T+tol
    for pt in tppoints:
        t=pt[0]
        if string=='low':
            if t<=bound:
                return pt # grab first instance (sorted by highest occuring minimize # TP points)
        elif string=='upper':
            if t>=bound:
                return pt
        
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


def WriteCSVFile(listoftpdics,nvtprops):
    firsttpdic=listoftpdics[0]
    firsttpdicvalues=list(firsttpdic.values())
    firsttppointvalues=firsttpdicvalues[0]
    firstglobaldic=firsttppointvalues['global']
    nvtproperties=['Surf_Ten']
    with open('data.csv', mode='w') as write_file:
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
                name=proptovalue['name']
                citation=proptovalue['cite']
                namels=['# Name',name]
                refls=['# Ref',citation]
                propls=['T','P']
                propvaluels=[str(tp[0]),str(tp[1])+' '+'atm']
                properrls=[]
                properrvaluels=[]
                commentpropvaluels=['# '+str(tp[0]),str(tp[1])+' '+'atm']

                ls=[namels,refls,commentpropvaluels]
                for prop,value in proptovalue.items():
                    if prop!='name' and prop!='cite' and 'wt' not in prop and 'err' not in prop and prop!='T' and prop!='P' and 'global' not in prop:
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
                        
def AddNistData(refstoproptoarray,propertynamearray,table,fbabbrtotargetunits):
    ls=[None for i in propertynamearray]
    if len(table)==0:
        return refstoproptoarray
    ureg = UnitRegistry()
    tableheader=table[0]
    nisttofbkeys={'vapH (kJ/mol)':'Hvap','Temperature (K)':'T','Cp (J/mol*K)':'Cp','Surf. Tension (N/m)':'Surf_Ten','Density (kg/m3)':'Rho'}
    if 'Reference' in tableheader:
        refidx=tableheader.index('Reference')
    else:
        refidx=None
    for row in table[1:]:
        for i in range(len(row)):
            tablehead=tableheader[i]
            value=row[i]
            if tablehead=='Phase':
                if value=='vapor': # skip, only want liquid
                    continue
            if tablehead in nisttofbkeys.keys():
                fbvalue=nisttofbkeys[tablehead]  
                if refidx!=None:
                    ref=row[refidx]
                else:
                    ref='Unknown'
                if tablehead=='Cp (J/mol*K)': # NIST doesnt allow kJ for this so just convert here
                   value=float(value)*.001
                if tablehead=='Surf. Tension (N/m)': # NIST doesnt allow desired units so convert here
                    currentunit='N/m'
                    targetunit=fbabbrtotargetunits[fbvalue]
                    string=value+' * '+currentunit+' to '+targetunit
                    src, dst = string.split(' to ')
                    Q_ = ureg.Quantity
                    convobj=Q_(src).to(dst)
                    convfactor=convobj._magnitude
                    converted=float(value)*convfactor 
                    value=converted
                dic=dict(zip(propertynamearray,ls))
                if ref not in refstoproptoarray.keys(): 
                    refstoproptoarray[ref]=dic
                if refstoproptoarray[ref][fbvalue]==None:
                    refstoproptoarray[ref][fbvalue]=[]
                refstoproptoarray[ref][fbvalue].append(value)

    return refstoproptoarray

def QueryNISTProperties(truename,names):
    fbabbrtotargetunits={'Eps0':'debye**2/(nm**3 .J)','Eps0_err':'debye**2/(nm**3 .J)','Cp':'kJ/(mol.K)','Cp_err':'kJ/(mol.K)','Kappa':'1/bar','Kappa_err':'1/bar','Alpha':'1/K','Alpha_err':'1/K','Surf_Ten':'mJ/m**2','Surf_Ten_err':'mJ/m**2','T':'K','P':'atm','Rho':'kg/m**3','Rho_err':'kg/m**3','Hvap':'kJ/mol','Hvap_err':'kJ/mol'}
    refstoproptoarray={}
    if truename!=None:
        x  = search_nist(name=truename)
    else:
        for tryname in names:
            try:
                x  = search_nist(name=tryname)
                break
            except:
                continue
        

    firstx=x[0]# # by default try first item in search results
    y = firstx.instantiate()
    temp=300
    try: # query error, no data available (sometimes not fluid data or just wrong temperature)
        table=y._get_fluid_data(firstx,temp)
    except:
        table=None
    phasechange=y._get_PhaseChange_data(firstx,temp)

    try:
        sattable=y._get_fluid_Saturation_data(firstx)
    except:
        sattable=None

    propertynamearray=list(fbabbrtotargetunits.keys())
    try:
        length=len(sattable)
        refstoproptoarray=AddNistData(refstoproptoarray,propertynamearray,sattable,fbabbrtotargetunits)
    except:
        pass
    try:
        length=len(table)
        refstoproptoarray=AddNistData(refstoproptoarray,propertynamearray,table,fbabbrtotargetunits)
    except:
        pass
    try:
        length=len(phasechange)
        refstoproptoarray=AddNistData(refstoproptoarray,propertynamearray,phasechange,fbabbrtotargetunits)
    except:
        pass

    
    return refstoproptoarray

def RemoveSamePoints(listoftpdics):
    newlistoftpdics=[]
    for tpdic in listoftpdics:
        if tpdic not in newlistoftpdics:
            newlistoftpdics.append(tpdic)

    return newlistoftpdics

def RemoveDictionariesWithNoExpDataAtTargetTP(listoftpdics):
    newlistoftpdics=[]
    for tpdic in listoftpdics:
        newdic={}
        for tppoint,dic in tpdic.items():
            founddata=False
            for key,value in dic.items():
                if key!='global' and key!='cite' and key!='name' and '_wt' not in key and '_denom' not in key:
                    if value!=None:
                        founddata=True
            if founddata==True:
                newdic[tppoint]=dic
        if len(newdic.keys())>0:
            newlistoftpdics.append(newdic)
            
    return newlistoftpdics


def ParseDatabaseGenerateLiquidCSVFile(smiles,truename,printnames,searchILDB,rep,roomtemp,lowertol,uppertol,nvtprops):

    names=[]
    if smiles!=None:
        names=ConvertSmilestoNames(smiles,rep)
    if printnames==True:
        print('names from pubchem',names)
        print('smiles',smiles)
    if searchILDB==True:
        refstoproptoarray=QueryLiquidProperties(names,truename,printnames)
    else:
        refstoproptoarray=QueryNISTProperties(truename,names)
    
    sortedrefstoproptoarray=SortReferences(refstoproptoarray,searchILDB)
    sortedrefstoproptoarray=RoundTPData(sortedrefstoproptoarray,2) # round to two decimal places
    proptorefs=PropertyToReferences(sortedrefstoproptoarray)
    sortedtemppressurecounts=CountTPFrequency(sortedrefstoproptoarray)
    tppoint=GrabNearestTPPoint(roomtemp,list(sortedtemppressurecounts.keys()))
    tptoproptovalue=FindMinimalTPPointsAllProps(tppoint,proptorefs,sortedrefstoproptoarray,truename)
    tptoproptovalue=AddDefaultValues(tptoproptovalue)
    lowertppoint=GrabTPPointOutsideBound(tppoint,lowertol,list(sortedtemppressurecounts.keys()),'low')
    uppertppoint=GrabTPPointOutsideBound(tppoint,uppertol,list(sortedtemppressurecounts.keys()),'upper')
    lowertptoproptovalue=FindMinimalTPPointsAllProps(lowertppoint,proptorefs,sortedrefstoproptoarray,truename)
    lowertptoproptovalue=AddDefaultValues(lowertptoproptovalue)
    uppertptoproptovalue=FindMinimalTPPointsAllProps(uppertppoint,proptorefs,sortedrefstoproptoarray,truename)
    uppertptoproptovalue=AddDefaultValues(uppertptoproptovalue)
    listoftpdics=[tptoproptovalue,lowertptoproptovalue,uppertptoproptovalue]
    listoftpdics=RemoveDictionariesWithNoExpDataAtTargetTP(listoftpdics)
    listoftpdics=RemoveSamePoints(listoftpdics)
    density=GrabDensity(listoftpdics)
    WriteCSVFile(listoftpdics,nvtprops)
    return density

def GrabDensity(listoftpdics):
    firstdic=listoftpdics[0]
    firstproptovaluekeylist=list(firstdic.keys())
    firstkey=firstproptovaluekeylist[0]
    proptovalue=firstdic[firstkey]
    density=float(proptovalue['Rho']) # in Kg/m**3 
    return density



def ReadInPoltypeFiles(poltypepath):
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
    return keyfile,xyzfile,dimertinkerxyzfiles,dimerenergies,molname,molfile
 

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


def GrabVdwTypeLinesFromFinalKey(keyfile,vdwtypes):
    vdwtypelines=[]
    temp=open(keyfile,'r')
    results=temp.readlines()
    temp.close()
    for line in results:
        if 'vdw' in line:
            for vdwtype in vdwtypes:
                if str(vdwtype) in line:
                    vdwtypelines.append(line)

    return vdwtypelines

def GenerateForceFieldFiles(vdwtypelines,moleculeprmfilename):
    if not os.path.isdir('forcefield'):
        os.mkdir('forcefield')
    os.chdir('forcefield')
    temp=open(moleculeprmfilename,'w')
    for line in vdwtypelines:
        linesplit=line.split()
        last=float(linesplit[-1])
        linelen=len(linesplit)
        linesplit.append('#')
        linesplit.append('PRM')
        linesplit.append('2')
        linesplit.append('3')
        if linelen==5 and last!=1:
            linesplit.append('4')
        newline=' '.join(linesplit)+'\n'
        temp.write(newline)
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


def GenerateLiquidTargetsFolder(gaskeyfile,gasxyzfile,liquidkeyfile,liquidxyzfile,datacsvpath,density,liquidfolder,prmfilepath,moleculeprmfilename,vdwtypes,addwaterprms):
    if not os.path.isdir('targets'):
        os.mkdir('targets')
    os.chdir('targets')
    if not os.path.isdir(liquidfolder):
        os.mkdir(liquidfolder)
    os.chdir(liquidfolder)
    shutil.copy(gaskeyfile,os.path.join(os.getcwd(),'gas.key'))
    RemoveKeyWord(os.path.join(os.getcwd(),'gas.key'),'parameters')
    temp=open(prmfilepath,'r')
    results=temp.readlines()
    temp.close()
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
    head,tail=os.path.split(liquidkeyfile)
    shutil.copy(liquidkeyfile,os.path.join(os.getcwd(),tail))
    CommentOutVdwLines(os.path.join(os.getcwd(),tail),vdwtypes)
    head,tail=os.path.split(liquidxyzfile)
    shutil.copy(liquidxyzfile,os.path.join(os.getcwd(),tail))
    os.remove(liquidxyzfile)
    if datacsvpath!=None:
        head,tail=os.path.split(datacsvpath)
        shutil.copy(datacsvpath,os.path.join(os.getcwd(),tail))
        os.remove(datacsvpath)
    os.chdir('..')
    os.chdir('..')


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

def CreateSolventBox(axis,molnumber,prmfilepath,xyzeditpath,tinkerxyzname):
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
    os.replace(tail+'_2','liquid.xyz') 
    liquidxyzfile=os.path.join(os.getcwd(),'liquid.xyz')
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

def GenerateNewKeyFile(keyfile,prmfilepath,moleculeprmfilename,axis,addwaterprms):
    liquidkeyfile=shutil.copy(keyfile,os.path.join(os.getcwd(),'liquid.key'))
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

def GenerateTargetFiles(keyfile,xyzfile,density,rdkitmol,prmfilepath,xyzeditpath,moleculeprmfilename,addwaterprms):
    mass=Descriptors.ExactMolWt(rdkitmol)*1.66054*10**(-27) # convert daltons to Kg
    axis=ComputeBoxLength(xyzfile)
    boxlength=axis*10**-10 # convert angstroms to m
    numbermolecules=int(density*boxlength**3/mass)
    liquidxyzfile=CreateSolventBox(axis,numbermolecules,prmfilepath,xyzeditpath,xyzfile)
    gaskeyfile=keyfile
    gasxyzfile=xyzfile
    if os.path.exists('data.csv'):
        datacsvpath=os.path.join(os.getcwd(),'data.csv')
    else:
        datacsvpath=None
    liquidkeyfile=GenerateNewKeyFile(keyfile,prmfilepath,moleculeprmfilename,axis,addwaterprms)

    return gaskeyfile,gasxyzfile,liquidkeyfile,liquidxyzfile,datacsvpath

def GenerateQMTargetsFolder(dimertinkerxyzfiles,dimerenergies,liquidkeyfile,qmfolder,vdwtypes):
    os.chdir('targets')
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
    head,tail=os.path.split(liquidkeyfile)
    shutil.copy(liquidkeyfile,os.path.join(os.getcwd(),tail))
    CommentOutVdwLines(os.path.join(os.getcwd(),tail),vdwtypes)
    os.remove(liquidkeyfile)
    os.chdir('..')
    os.chdir('..') 

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


def ConvertInputMoleculeToSmiles(molfile):
    rdkitmol=GenerateRdkitMol(molfile)
    smiles=Chem.MolToSmiles(rdkitmol) 
    return smiles


def GenerateForceBalanceInputFile(moleculeprmfilename,qmfolder,liquidfolder,optimizefilepath,atomnum,liquid_equ_steps,liquid_prod_steps,liquid_timestep,liquid_interval,gas_equ_steps,gas_timestep,gas_interval,md_threads):
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
    
    results.append('$target'+'\n')
    results.append('name '+qmfolder+'\n')
    results.append('type Interaction_TINKER'+'\n')
    results.append('weight 1.0'+'\n')
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

if poltypepath!=None:
    keyfile,xyzfile,dimertinkerxyzfiles,dimerenergies,molname,molfile=ReadInPoltypeFiles(poltypepath)
    smiles=ConvertInputMoleculeToSmiles(molfile)

if skipNIST==False:
    density=ParseDatabaseGenerateLiquidCSVFile(smiles,truename,printnames,searchILDB,rep,roomtemp,lowertol,uppertol,nvtprops)
if poltypepath!=None:
    prmfilepath=os.path.join(os.path.split(__file__)[0],'amoebabio18.prm')
    optimizefilepath=os.path.join(os.path.split(__file__)[0],'optimize.in')
    xyzeditpath='xyzedit'
    tinkerdir=None
    xyzeditpath=SanitizeMMExecutable(xyzeditpath,tinkerdir)
    vdwtypelines=GrabVdwTypeLinesFromFinalKey(keyfile,vdwtypes)
    moleculeprmfilename='molecule.prm'
    GenerateForceFieldFiles(vdwtypelines,moleculeprmfilename)
    rdkitmol=GenerateRdkitMol(molfile)
    atomnum=rdkitmol.GetNumAtoms()
    gaskeyfile,gasxyzfile,liquidkeyfile,liquidxyzfile,datacsvpath=GenerateTargetFiles(keyfile,xyzfile,density,rdkitmol,prmfilepath,xyzeditpath,moleculeprmfilename,addwaterprms)
    liquidfolder='Liquid'
    qmfolder='QM'
    GenerateLiquidTargetsFolder(gaskeyfile,gasxyzfile,liquidkeyfile,liquidxyzfile,datacsvpath,density,liquidfolder,prmfilepath,moleculeprmfilename,vdwtypes,addwaterprms)
    GenerateQMTargetsFolder(dimertinkerxyzfiles,dimerenergies,liquidkeyfile,qmfolder,vdwtypes)    
    GenerateForceBalanceInputFile(moleculeprmfilename,qmfolder,liquidfolder,optimizefilepath,atomnum,liquid_equ_steps,liquid_prod_steps,liquid_timestep,liquid_interval,gas_equ_steps,gas_timestep,gas_interval,md_threads) 
