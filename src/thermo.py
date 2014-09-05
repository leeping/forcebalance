import os
import errno
import numpy as np

from forcebalance.target import Target
from forcebalance.finite_difference import in_fd, f12d3p, fdwrap
from forcebalance.nifty import flat, col, row
from forcebalance.nifty import lp_dump, lp_load, wopen, _exec
from forcebalance.nifty import LinkFile, link_dir_contents
from forcebalance.nifty import printcool, printcool_dictionary

from collections import OrderedDict

from forcebalance.output import getLogger
logger = getLogger(__name__)

def energy_derivatives(engine, FF, mvals, h, pgrad, dipole=False):

    """
    Compute the first and second derivatives of a set of snapshot
    energies with respect to the force field parameters.

    This basically calls the finite difference subroutine on the
    energy_driver subroutine also in this script.

    In the future we may need to be more sophisticated with
    controlling the quantities which are differentiated, but for
    now this is okay..

    @param[in] engine Engine object for calculating energies
    @param[in] FF Force field object
    @param[in] mvals Mathematical parameter values
    @param[in] h Finite difference step size
    @param[in] pgrad List of active parameters for differentiation
    @param[in] dipole Switch for dipole derivatives.
    @return G First derivative of the energies in a N_param x N_coord array
    @return GDx First derivative of the box dipole moment x-component in a N_param x N_coord array
    @return GDy First derivative of the box dipole moment y-component in a N_param x N_coord array
    @return GDz First derivative of the box dipole moment z-component in a N_param x N_coord array

    """
    def single_point(mvals_):
        FF.make(mvals_)
        if dipole:
            return engine.energy_dipole()
        else:
            return engine.energy()

    ED0 = single_point(mvals)
    G   = OrderedDict()
    G['potential'] = np.zeros((FF.np, ED0.shape[0]))
    if dipole:
        G['dipole'] = np.zeros((FF.np, ED0.shape[0], 3))
    for i in pgrad:
        logger.info("%i %s\r" % (i, (FF.plist[i] + " "*30)))
        edg, _ = f12d3p(fdwrap(single_point,mvals,i),h,f0=ED0)
        if dipole:
            G['potential'][i] = edg[:,0]
            G['dipole'][i]    = edg[:,1:]
        else:
            G['potential'][i] = edg[:]
    return G

#
class Thermo(Target):
    """
    A target for fitting general experimental data sets. The
    experimental data is described in a .txt file and is handled with a
    `Quantity` subclass.

    """
    def __init__(self, options, tgt_opts, forcefield):
        ## Initialize base class
        super(Thermo, self).__init__(options, tgt_opts, forcefield)

        ## Parameters
        # Reference experimental data
        self.set_option(tgt_opts, "expdata_txt", forceprint=True)
        # Quantities to calculate
        self.set_option(tgt_opts, "quantities", forceprint=True)
        # Length of simulation chain
        self.set_option(tgt_opts, "n_sim_chain", forceprint=True)
        # Number of time steps in the equilibration run
        self.set_option(tgt_opts, "eq_steps", forceprint=True)
        # Number of time steps in the production run
        self.set_option(tgt_opts, "md_steps", forceprint=True)

        ## Variables
        # Prefix names for simulation data
        self.simpfx    = "sim"
        # Data points for quantities
        self.points    = []
        # Denominators for quantities
        self.denoms    = {}
        # Weights for quantities
        self.weights   = {}

        ## Read experimental data and initialize points
        self._read_expdata(os.path.join(self.root,
                                        self.tgtdir,
                                        self.expdata_txt))
        
        ## Copy run scripts from ForceBalance installation directory
        for f in self.scripts:
            LinkFile(os.path.join(os.path.split(__file__)[0], "data", f),
                     os.path.join(self.root, self.tempdir, f))
    
    def _read_expdata(self, expdata):
        """Read and store experimental data.

        Parameters
        ----------
        expdata : string
            Read experimental data from this filename.

        Returns
        -------
        Nothing

        """
        fp = open(expdata)

        line         = fp.readline()
        foundHeader  = False
        names        = None
        units        = None
        label_header = None
        label_unit   = None
        count        = 0
        while line:
            # Skip comments and blank lines
            if line.lstrip().startswith("#") or not line.strip():
                line = fp.readline()
                continue

            if "=" in line: # Read variable
                param, value = line.split("=")
                param = param.strip().lower()
                if param == "denoms":
                    for e, v in enumerate(value.split()):
                        self.denoms[self.quantities[e]] = float(v)
                elif param == "weights":
                    for e, v in enumerate(value.split()):
                        self.weights[self.quantities[e]] = float(v)
            elif foundHeader: # Read exp data
                count      += 1
                vals        = line.split()
                
                label       = (vals[0], label_header, label_unit)
                refs        = np.array(vals[1:-2:2]).astype(float)
                wts         = np.array(vals[2:-2:2]).astype(float)
                temperature = float(vals[-2])
                pressure    = None if vals[-1].lower() == "none" else \
                  float(vals[-1])
                
                dp = Point(count, label=label, refs=refs, weights=wts,
                           names=names, units=units,
                           temperature=temperature, pressure=pressure)
                self.points.append(dp)
            else: # Read headers
                foundHeader = True
                headers = zip(*[tuple(h.split("_")) for h in line.split()
                                if h != "w"])

                label_header = list(headers[0])[0]
                label_unit   = list(headers[1])[0]
                names        = list(headers[0][1:-2])
                units        = list(headers[1][1:-2])
                                
            line = fp.readline()            
    
    def retrieve(self, dp):
        """Retrieve the molecular dynamics (MD) results and store the calculated
        quantities in the Point object dp.

        Parameters
        ----------
        dp : Point
            Store the calculated quantities in this point.

        Returns
        -------
        Nothing
        
        """
        abspath = os.path.join(os.getcwd(), '%d/md_result.p' % dp.idnr)

        if os.path.exists(abspath):
            logger.info('Reading data from ' + abspath + '.\n')

            vals, errs, grads = lp_load(abspath)

            dp.data["values"] = vals
            dp.data["errors"] = errs
            dp.data["grads"]  = grads

        else:
            msg = 'The file ' + abspath + ' does not exist so we cannot read it.\n'
            logger.warning(msg)

            dp.data["values"] = np.zeros((len(self.quantities)))
            dp.data["errors"] = np.zeros((len(self.quantities)))
            dp.data["grads"]  = np.zeros((len(self.quantities), self.FF.np))
            
    def submit_jobs(self, mvals, AGrad=True, AHess=True):
        """This routine is called by Objective.stage() and will run before "get".
        It submits the jobs and the stage() function will wait for jobs
        to complete.

        Parameters
        ----------
        mvals : list
            Mathematical parameter values.
        AGrad : Boolean
            Switch to turn on analytic gradient.
        AHess : Boolean
            Switch to turn on analytic Hessian.

        Returns
        -------
        Nothing.
        
        """
        # Set up and run the simulation chain on all points.
        for pt in self.points:
            # Create subdir
            try:
                os.makedirs(str(pt.idnr))
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise            
                
            # Goto subdir
            os.chdir(str(pt.idnr))

            # Link dir contents from target subdir to current temp directory.
            for f in self.scripts:
                LinkFile(os.path.join(self.root, self.tempdir, f),
                         os.path.join(os.getcwd(), f))
                
            link_dir_contents(os.path.join(self.root, self.tgtdir,
                                           str(pt.idnr)), os.getcwd())
            
            # Dump the force field to a pickle file
            lp_dump((self.FF, mvals, self.OptionDict, AGrad), 'forcebalance.p')
                
            # Run the simulation chain for point.        
            cmdstr = ("%s python md_chain.py " % self.mdpfx +
                      " ".join(self.quantities) + " " +
                      "--engine %s " % self.engname +
                      "--length %d " % self.n_sim_chain + 
                      "--name %s " % self.simpfx +
                      "--temperature %f " % pt.temperature +
                      "--pressure %f " % pt.pressure +
                      "--nequil %d " % self.eq_steps +
                      "--nsteps %d " % self.md_steps)
            _exec(cmdstr, copy_stderr=True, outfnm='md_chain.out')
        
            os.chdir('..')

    def indicate(self):
        """Shows optimization state."""
        AGrad     = hasattr(self, 'Gp')
        PrintDict = OrderedDict()
        
        def print_item(key, physunit):
            if self.Xp[key] > 0:
                the_title = ("%s %s (%s)\n" % (self.name, key.capitalize(), physunit) +
                             "No.   Temperature  Pressure  Reference  " +
                             "Calculated +- Stddev " +
                             "   Delta    Weight    Term   ")
                    
                printcool_dictionary(self.Pp[key], title=the_title, bold=True,
                                     color=4, keywidth=15)
                
                bar = printcool(("%s objective function: % .3f%s" %
                                 (key.capitalize(), self.Xp[key],
                                  ", Derivative:" if AGrad else "")))
                if AGrad:
                    self.FF.print_map(vals=self.Gp[key])
                    logger.info(bar)

                PrintDict[key] = (("% 10.5f % 8.3f % 14.5e" %
                                   (self.Xp[key], self.Wp[key],
                                    self.Xp[key]*self.Wp[key])))

        for i, q in enumerate(self.quantities):
            print_item(q, self.points[0].ref["units"][i])

        PrintDict['Total'] = "% 10s % 8s % 14.5e" % ("","", self.Objective)

        Title = ("%s Thermodynamic Properties:\n %-20s %40s" %
                 (self.name, "Property", "Residual x Weight = Contribution"))
        printcool_dictionary(PrintDict, color=4, title=Title, keywidth=31)
        return

    def objective_term(self, quantity):
        """Calculates the contribution to the objective function (the term) for a
        given quantity.

        Parameters
        ----------
        quantity : string
            Calculate the objective term for this quantity.

        Returns
        -------
        term : dict
            `term` is a dict with keys `X`, `G`, `H` and `info`. The values of
            these keys are the objective term itself (`X`), its gradient (`G`),
            its Hessian (`H`), and an OrderedDict with print information on
            individiual data points (`info`).
            
        """
        Objective = 0.0
        Gradient  = np.zeros(self.FF.np)
        Hessian   = np.zeros((self.FF.np, self.FF.np))

        # Grab ref data for quantity        
        qid       = self.quantities.index(quantity)
        Exp       = np.array([pt.ref["refs"][qid] for pt in self.points])
        Weights   = np.array([pt.ref["weights"][qid] for pt in self.points])
        Denom     = self.denoms[quantity]
            
        # Renormalize weights
        Weights /= np.sum(Weights)
        logger.info("Renormalized weights to " + str(np.sum(Weights)) + "\n")
        logger.info(("Physical quantity '%s' uses denominator = %g %s\n" %
                     (quantity.capitalize(), Denom,
                      self.points[0].ref["units"][self.quantities.index(quantity)])))

        # Grab calculated values        
        values = np.array([pt.data["values"][qid] for pt in self.points])
        errors = np.array([pt.data["errors"][qid] for pt in self.points])
        grads  = np.array([pt.data["grads"][qid] for pt in self.points])

        # Calculate objective term using Least-squares function. Evaluate using
        # Einstein summation: W is N-array, Delta is N-array and grads is
        # NxM-array, where N is number of points and M is number of parameters.
        #
        #     X_i   = W_i * Delta2_i (no summed indices)
        #     G_ij  = W_i * Delta_i * grads_ij (no summed indices)
        #     H_ijm = W_i * gradsT_jk * grads_lm (sum over k and l)
        #
        # Result: X is N-array, G is NxM-array and H is NxMxM-array.
        #
        Deltas = values - Exp
        Objs   = np.einsum("i,i->i", Weights, Deltas**2) / Denom / Denom
        Grads  = 2.0*np.einsum("i,i,ij->ij", Weights, Deltas, grads) / Denom / Denom
        Hess   = 2.0*np.einsum("i,jk,lm->ijm", Weights, grads.T, grads) / Denom / Denom
        
        # Average over all points
        Objective += np.sum(Objs, axis=0)
        Gradient  += np.sum(Grads, axis=0)
        Hessian   += np.sum(Hess, axis=0)
        
        # Store gradients and setup print map 
        GradMapPrint = [["#Point"] + self.FF.plist]

        for pt in self.points:
            temp  = pt.temperature
            press = pt.pressure
            GradMapPrint.append([' %8.2f %8.1f' % (temp, press)] +
                                ["% 9.3e" % i for i in grads[pt.idnr-1]])

        o = wopen('gradient_%s.dat' % quantity)
        for line in GradMapPrint:
            print >> o, ' '.join(line)
        o.close()
        
        printer = OrderedDict([("    %-5d %-12.2f %-8.1f"
                                % (pt.idnr, pt.temperature, pt.pressure),
                                ("% -10.3f % -10.3f  +- %-8.3f % -8.3f % -9.5f % -9.5f"
                                 % (Exp[pt.idnr-1], values[pt.idnr-1],
                                    errors[pt.idnr-1], Deltas[pt.idnr-1],
                                    Weights[pt.idnr-1], Objs[pt.idnr-1])))
                                    for pt in self.points])
                
        return { "X": Objective, "G": Gradient, "H": Hessian, "info": printer }

    def get(self, mvals, AGrad=True, AHess=True):
        """Return the contribution to the total objective function. This is a
        weighted average of the calculated quantities.

        Parameters
        ----------
        mvals : list
            Mathematical parameter values.
        AGrad : Boolean
            Switch to turn on analytic gradient.
        AHess : Boolean
            Switch to turn on analytic Hessian.

        Returns
        -------
        Answer : dict
            Contribution to the objective function. `Answer` is a dict with keys
            `X` for the objective function, `G` for its gradient and `H` for its
            Hessian.
                    
        """
        Answer   = {}

        Objective = 0.0
        Gradient  = np.zeros(self.FF.np)
        Hessian   = np.zeros((self.FF.np, self.FF.np))

        for pt in self.points:
            # Update data point with MD results
            self.retrieve(pt)

        obj        = OrderedDict()
        reweighted = []
        for q in self.quantities:
            # Returns dict with keys "X"=objective term value, "G"=the
            # gradient, "H"=the hessian, and "info"=printed info about points
            obj[q] = self.objective_term(q)
        
            # Apply weights for quantities (normalized)
            if obj[q]["X"] == 0:
                self.weights[q] = 0.0

            # Store weights sorted in the order of self.quantities
            reweighted.append(self.weights[q])
        
        # Normalize weights
        reweighted  = np.array(reweighted)
        wtot        = np.sum(reweighted)
        reweighted  = reweighted/wtot if wtot > 0 else reweighted
         
        # Picks out the "X", "G" and "H" keys for the quantities sorted in the
        # order of self.quantities. Xs is N-array, Gs is NxM-array and Hs is
        # NxMxM-array, where N is number of quantities and M is number of
        # parameters.
        Xs = np.array([dic["X"] for dic in obj.values()])
        Gs = np.array([dic["G"] for dic in obj.values()])
        Hs = np.array([dic["H"] for dic in obj.values()])
                                
        # Target contribution is (normalized) weighted averages of the
        # individual quantity terms.
        Objective    = np.average(Xs, weights=(None if np.all(reweighted == 0) else \
                                               reweighted), axis=0)
        if AGrad:
            Gradient = np.average(Gs, weights=(None if np.all(reweighted == 0) else \
                                               reweighted), axis=0)
        if AHess:
            Hessian  = np.average(Hs, weights=(None if np.all(reweighted == 0) else \
                                               reweighted), axis=0)

        if not in_fd():
            # Store results to show with indicator() function
            self.Xp = {q : dic["X"] for (q, dic) in obj.items()}
            self.Wp = {q : reweighted[self.quantities.index(q)]
                       for (q, dic) in obj.items()}
            self.Pp = {q : dic["info"] for (q, dic) in obj.items()}

            if AGrad:
                self.Gp = {q : dic["G"] for (q, dic) in obj.items()}

            self.Objective = Objective
        
        Answer = { "X": Objective, "G": Gradient, "H": Hessian }
        return Answer
    
# class Point --- data container
class Point(object):
    def __init__(self, idnr, label=None, refs=None, weights=None, names=None,
                 units=None, temperature=None, pressure=None, data=None):
        self.idnr        = idnr
        self.ref         = { "label"  : label,                    
                             "refs"   : refs,
                             "weights": weights,
                             "names"  : names,
                             "units"  : units }
        self.temperature = temperature
        self.pressure    = pressure
        self.data        = data if data is not None else {}
        
    def __str__(self):
        msg = []
        if self.temperature is None:
            msg.append("State: Unknown.")
        elif self.pressure is None:
            msg.append("State: Point " + str(self.idnr) + " at " +
                       str(self.temperature) + " K.")
        else:
            msg.append("State: Point " + str(self.idnr) + " at " +
                       str(self.temperature) + " K and " +
                       str(self.pressure) + " bar.")

        msg.append("Point " + str(self.idnr) + " reference data " + "-"*30)
        for key in self.ref:
            msg.append("  " + key.strip() + " = " + str(self.ref[key]).strip())
            
        msg.append("Point " + str(self.idnr) + " calculated data " + "-"*30)
        for key in self.data:
            msg.append("  " + key.strip() + " = " + str(self.data[key]).strip())

        return "\n".join(msg)
