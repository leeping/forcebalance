import os, numpy as np
import pyilt2
import sys
import cirpy
import pint
from pint import UnitRegistry
import csv
import getopt
from tqdm import tqdm


from bs4 import BeautifulSoup # for parsing NIST
import requests
import re
import pandas as pd
import numpy as np

'''
conda create --name FBTest --yes
conda activate FBTest
conda install pip --yes
pip install pyilt2
conda install -c conda-forge cirpy --yes
conda install -c conda-forge pint --yes
conda install -c conda-forge tqdm --yes
conda install -c anaconda beautifulsoup4 --yes
conda install -c anaconda pandas --yes


'''

# https://ilthermo.boulder.nist.gov/
# Be careful by default tries first name it finds with molecule match, but may not be correct name. Use printnames to find possible names that are found in database. Use truename to only grab data from the correct molecule name (according to the database)

# python gentargetcsv.py --smiles=C(Cl)(Cl)(Cl)Cl
# python gentargetcsv.py --smiles=C(Cl)(Cl)(Cl)Cl --printnames
# python gentargetcsv.py --smiles=C(Cl)(Cl)(Cl)Cl --truename=Tetrachloromethane
# python gentargetcsv.py --truename=Water

smiles=None
truename=None
printnames=False
searchILDB=False # search NIST by default

opts, xargs = getopt.getopt(sys.argv[1:],'',["smiles=","truename=",'printnames','searchILDB'])
for o, a in opts:
    if o in ("--smiles"):
        smiles=a
    elif o in ("--truename"):
        truename=a.capitalize() # the database has first letter capitalized 
    elif o in ("--printnames"):
        printnames=True
    elif o in ("--searchILDB"):
        searchILDB=False

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


def WriteCSVFile(listoftpdics):
    firsttpdic=listoftpdics[0]
    firsttpdicvalues=list(firsttpdic.values())
    firsttppointvalues=firsttpdicvalues[0]
    firstglobaldic=firsttppointvalues['global']

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

def QueryNISTProperties(truename):
    fbabbrtotargetunits={'Eps0':'debye**2/(nm**3 .J)','Eps0_err':'debye**2/(nm**3 .J)','Cp':'kJ/(mol.K)','Cp_err':'kJ/(mol.K)','Kappa':'1/bar','Kappa_err':'1/bar','Alpha':'1/K','Alpha_err':'1/K','Surf_Ten':'mJ/m**2','Surf_Ten_err':'mJ/m**2','T':'K','P':'atm','Rho':'kg/m**3','Rho_err':'kg/m**3','Hvap':'kJ/mol','Hvap_err':'kJ/mol'}
    refstoproptoarray={}
    print('truename',truename)
    x  = search_nist(name=truename)
    print('x',x)
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

rep='names'
roomtemp=295.5
lowertol=10
uppertol=10
names=[]
if smiles!=None:
    names=ConvertSmilestoNames(smiles,rep)
if printnames==True:
    print('names from pubchem',names)
if searchILDB==True:
    refstoproptoarray=QueryLiquidProperties(names,truename,printnames)
else:
    if truename==None: # then take first (might be wrong please check)
        truename=names[0]
    refstoproptoarray=QueryNISTProperties(truename)

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
WriteCSVFile(listoftpdics)
