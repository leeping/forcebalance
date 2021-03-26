#!/usr/bin/env python
import os, sys, shutil, re
from forcebalance.molecule import *
from forcebalance.forcefield import FF
from forcebalance.objective import Implemented_Targets
from forcebalance.nifty import _exec, printcool
from forcebalance.parser import parse_inputs
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import argparse

def getCentroid(r_rmsd, beta=1.0):
    """
    Gets index of the centroid of a cluster. Follows previous MDTraj procedure.
    http://mdtraj.org/1.4.2/examples/centroids.html

    Parameters
    -----------
    r_rmsd : np.ndarray
        Pairwise RMSD distances used for obtaining centroids
    beta : float, optinal
        parameter used in computing the similarity score below
        default set to 1.0

    Returns
    -----------
    int
        If cluster only has one member, then return an index of 0. Otherwise,
        return the maximum of the computed similarity score.
    """
    if len(r_rmsd) > 1: return np.exp(-beta*r_rmsd/r_rmsd.std()).sum(axis=1).argmax()
    else: return 0

class Reopt:
    """
    This class handles the main functions of the code, including the clustering.
    The inputs are kwargs taken from the argparse input.
    """
    def __init__(self, **kwargs):
        #Get all of the user info needed for the code.

        #Need a ForceBalance directory for the targets and the MM options.
        self.fbinput = kwargs.get('fbinput')
        
        #Determine whether to use heavy atoms or all atoms for clustering
        self.cluster_inds = kwargs.get("default_cluster_indices")
        #Distance cutoff used for hierarchial clustering
        self.clust_dist = kwargs.get("cluster_dist")

        self.base_tmp = "reopt.tmp"
        self.opt_name = "opt"
        self.min_file_name = "min.gro"

    def parseFBInput(self):
        """
        This reads through the provided ForceBalance input file using the 
        standard FB parse_inputs. It removes any non-AbInitio targets and 
        removes any AbInitio targets that are previously MMOpt targets.
        This forms a dictionary (self.unique_res) containing targets belonging
        to the same residue based off of the target prefix.
        """
        printcool("Reading Grids")
        
        #Parse FB input file
        self.options, self.tgt_opts = parse_inputs(self.fbinput)

        #Get force field in FB result directory 
        ff_path = os.path.join("result", os.path.splitext(self.options["input_file"])[0])
        self.options["ffdir"] = ff_path
        self.ff = FF(self.options)

        #Retain AbInitio targets that are not mmopt targets
        self.tgt_opts = [l for l in self.tgt_opts if "ABINITIO" in l.get("type") and "mmopt" not in l.get("name")]

        self.root = self.options["root"]

        self.options["input_file"] = "reopt" 

        #Assemble targets from ImplementedTargets dictionary
        self.targets = []
        for opts in self.tgt_opts:
            Tgt = Implemented_Targets[opts["type"]](self.options, opts, self.ff)
            self.targets.append(Tgt)

        #Combine targets that belong to one residue, splits on - or _ in target name (may not be completely sufficient...)
        self.unique_res = {}
        for i in range(len(self.tgt_opts)):
            name = re.split(r"_|-", self.tgt_opts[i]["name"])[0]
            if name in self.unique_res: self.unique_res[name].append(self.targets[i])
            else:
                self.unique_res[name] = []
                self.unique_res[name].append(self.targets[i])

    def minGrids(self):
        """
        For each unique residue, loop through the engine.mol object of each target and minimize all structures.
        Then gather all the files for each target into one file that is placed in the "Cluster" directory. 
        """
        printcool("Initial MM Minimization")

        cwd = os.getcwd()
        if not os.path.isdir("{}/Cluster".format(self.base_tmp)): os.makedirs("{}/Cluster".format(self.base_tmp))
        for k in self.unique_res:
            collect = None
            for i in range(len(self.unique_res[k])):
                os.chdir(self.unique_res[k][i].tempdir)
                scr = os.path.join(self.root, self.unique_res[k][i].tempdir)
                if not os.path.isdir("{}".format(self.opt_name)): os.makedirs("{}".format(self.opt_name))
                os.chdir("{}".format(self.opt_name))
                for struct in range(len(self.unique_res[k][i].engine.mol)):
                    if not os.path.isdir("{}".format(struct)): os.makedirs("{}".format(struct))
                    os.chdir("{}".format(struct))
                    engine_files = os.listdir("{}".format(scr))
                    [os.symlink("{}/{}".format(scr,z), z) for z in engine_files if not os.path.isdir("{}/{}".format(scr,z))]
                    os.symlink(os.path.join(self.root,self.options['ffdir'],self.options['forcefield'][0]), "{}".format(self.options['forcefield'][0]))
                    energy, rmsd, geom = self.unique_res[k][i].engine.optimize(struct)
                    geom.write("{}".format(self.min_file_name))
                    os.chdir(os.path.join(scr, self.opt_name))
                _exec('find . -name {} | sort | xargs cat > mm_opt.gro'.format(self.min_file_name))
                mol = Molecule("mm_opt.gro")
                if collect is None: collect = mol
                else: collect.append(mol)
                os.chdir(cwd)
            os.chdir("{}/Cluster".format(self.base_tmp))
            if not os.path.isdir(k): os.makedirs(k)
            os.chdir(k)
            collect.write("mm_opt.gro")
            os.chdir(cwd)

    def Cluster(self):
        """
        Cluster structures for each residue using hierarchial clustering in scipy. Follows previous MDTraj procedure.
        http://mdtraj.org/1.4.2/examples/clustering.html
        """
        printcool("Clustering")
        cwd = os.getcwd()
        self.cluster_defs = {}
        for k in self.unique_res:
            os.chdir("{}/Cluster/{}".format(self.base_tmp,k))
            if not os.path.isfile("centroids.gro"):

                mm_opt = Molecule("mm_opt.gro")
                elems = mm_opt.elem

                if self.cluster_inds == "heavy": self.indices = [i for i in range(len(elems)) if elems[i]!="H"]
                elif self.cluster_inds == "all": self.indices = [i for i in range(len(elems))]
                else: raise Exception("{} is an invalid option for default_cluster_indices.".format(self.default_cluster_indices))
           
                distances = mm_opt.all_pairwise_rmsd(atom_inds=self.indices)

                reduced_distances = squareform(distances, checks=False)
                link = hierarchy.linkage(reduced_distances, method="centroid")
                clusters = hierarchy.fcluster(link, t=self.clust_dist, criterion="distance")

                self.cluster_defs = {}
                for i in clusters: self.cluster_defs[i] = []
                for i in range(len(clusters)): self.cluster_defs[clusters[i]].append(i)
            
                centroids = []
                for i in self.cluster_defs.keys():
                    clust_list = self.cluster_defs[i]
                    dist_clust = distances[clust_list, :][:, clust_list]
                    ind = getCentroid(dist_clust)
                    centroids.append(clust_list[ind])
                centroids_mmopt = mm_opt[centroids]
                centroids_mmopt.write("centroids.gro")
            os.chdir(cwd)
    
    def minCentroids(self):
        """
        MM optimize cluster centroids, final output is a separate directory for each structure.
        """
        printcool("Centroid Minimization")
        cwd = os.getcwd()
        for k in self.unique_res:
            os.chdir("{}/Cluster/{}".format(self.base_tmp,k))
            scr = os.path.join(self.root, self.unique_res[k][0].tempdir)
            centroids = Molecule("centroids.gro")
            self.unique_res[k][0].engine.mol = centroids
            for struct in range(len(self.unique_res[k][0].engine.mol)):
                if not os.path.isdir("{}".format(struct)): os.makedirs("{}".format(struct))
                os.chdir("{}".format(struct))
                if not os.path.isfile("{}".format(self.min_file_name)):
                    for f in os.listdir("."): os.remove(f)
                    engine_files = os.listdir("{}".format(scr))
                    [os.symlink("{}/{}".format(scr,z), z) for z in engine_files if not os.path.isdir("{}/{}".format(scr,z))]
                    os.symlink(os.path.join(self.root,self.options['ffdir'],self.options['forcefield'][0]), "{}".format(self.options['forcefield'][0]))
                    energy, rmsd, geom = self.unique_res[k][0].engine.optimize(struct)
                    geom.write("{}".format(self.min_file_name))
                os.chdir("{}".format(os.path.join(self.root, self.base_tmp, 'Cluster', k)))

            os.chdir(cwd)

def run_reopt(**kwargs):
    """
    Function for running all the code components.
    """

    reopt = Reopt(**kwargs)
    reopt.parseFBInput()
    reopt.minGrids()
    reopt.Cluster()
    reopt.minCentroids()

def main():
    parser = argparse.ArgumentParser(description="For use in conjunction with ForceBalance to identify spurious minima in the MM potential energy surface. The outputs from this program can then be added as ForceBalance targets for another optimization cycle.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fbinput', type=str, help="The name of the ForceBalance optimization input file.")
    parser.add_argument('--default_cluster_indices', type=str, default='heavy', help="The default clustering code will only pick out heavy atoms. Write 'all' if you want all indices to be used.")
    parser.add_argument('--cluster_dist', type=float, default=0.1, help="Distance criterion to be used for forming the hierarchical clusters")

    args = parser.parse_args()
    run_reopt(**vars(args))

if __name__ == "__main__":
    main()
