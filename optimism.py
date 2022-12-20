"""
optimism.py

OPTIMisation Including Surrogate Modelling

Library of optimisation functions.

(c) copyright Aleksander J Dubas 2021
Licensed under GPLv3, see LICENSE.
"""
import numpy as np
import numpy.linalg as npla


# surrogate modelling classes
class Surrogate():
    """
    Base class for a surrogate model.
    """
    def __init__(self):
        self.xs = []
        self.ys = []

    def loaddata(self, filename, noy=False):
        """
        Loads data into the surrogate model.

        Uses the same data structure as the savedata function.
        Use of an absolute path is recommended.

        Parameters
        ----------
        filename:   string
            Name of file to load data from.
        noy:    bool    (default False)
            Set to true if no y-data in file.

        Returns
        -------
        None
        """
        with open(filename, 'r') as fin:
            lines = fin.readlines()
        if noy:
            xlen = len(lines[0].split())
        else:
            xlen = len(lines[0].split()) - 1
        for line in lines:
            parts = line.split()
            self.xs.append([])
            for i in range(len(parts)):
                if i < xlen:
                    self.xs[-1].append(float(parts[i]))
                else:
                    self.ys.append(float(parts[i]))

        # convert xs to array
        self.xs = np.array(self.xs)
        return None

    def savedata(self, filename):
        """
        Saves data from a surrogate model.

        In the structure:
        xs[0][0] xs[0][1] xs[0][2] ... xs[0][-2] xs[0][-1] ys[0]\n
        xs[1][0] xs[1][1] xs[1][2] ... xs[1][-2] xs[1][-1] ys[1]\n
        Use of an absolute path is recommended.

        Parameters
        ----------
        filename:   string
            Name of file to save data to.

        Returns
        -------
        None
        """
        with open(filename, 'w') as fout:
            for i in range(len(self.xs)):
                for x in self.xs[i]:
                    fout.write(str(x)+" ")
                try:
                    fout.write(str(self.ys[i])+"\n")
                except IndexError:
                    fout.write("\n")

        return None

    def infill(self, f):
        """
        Evaluates any unevaluated points in self.xs array using f.

        Parameters
        ----------
        f:  function
            Function used to evaluate infill points.

        Returns
        -------
        None
        """
        xlen = len(self.xs)
        ylen = len(self.ys)
        if xlen == ylen:
            return None
        self.ys = np.hstack((self.ys, np.zeros(xlen-ylen)))
        for i in range(ylen, xlen):
            self.ys[i] = f(self.xs[i])
        return None

    def infill_point(self, x):
        """
        Adds infill point x to self.xs array.

        Parameters
        ----------
        x:  1-d array
            Infill point to be added.

        Returns
        -------
        None
        """
        self.xs = np.vstack((self.xs, x))
        return None

    def minx(self):
        """
        Returns the x value of the minimum (real) point.
        """
        return self.xs[np.argmin(self.ys)]

    def miny(self):
        """
        Returns the y value of the minimum (real) point.
        """
        return min(self.ys)

    def mini(self):
        """
        Returns the i value of the minimum (real) point.
        """
        return np.argmin(self.ys)

    def maxx(self):
        """
        Returns the x value of the maximum (real) point.
        """
        return self.xs[np.argmax(self.ys)]

    def maxy(self):
        """
        Returns the y value of the maximum (real) point.
        """
        return max(self.ys)

    def maxi(self):
        """
        Returns the i value of the maximum (real) point.
        """
        return np.argmax(self.ys)


class GaussianRBF(Surrogate):
    """
    Constructs a surrogate model using the Radial Basis Function in SciPy.
    Uses a Gaussian distribution.
    """
    def build(self):
        """
        Builds the surrogate model.
        """
        self.name = "GaussianRBF"
        from scipy.interpolate import Rbf

        # creating packing list for passing to RBF
        packinglist = []
        for i in range(len(self.xs[0])):
            packinglist.append(self.xs[:, i])
        packinglist.append(self.ys)
        self.rbf = Rbf(*tuple(packinglist), function="gaussian")

    def f(self, xs):
        """
        Evaluates the surrogate model at xs.
        """
        return self.rbf(*tuple(xs))


class MultiQuadricRBF(Surrogate):
    """
    Constructs a surrogate model using the Radial Basis Function in SciPy.
    Uses a MultiQuadric distribution.
    """
    def build(self):
        """
        Builds the surrogate model.
        """
        self.name = "MultiQuadricRBF"
        from scipy.interpolate import Rbf

        # creating packing list for passing to RBF
        packinglist = []
        for i in range(len(self.xs[0])):
            packinglist.append(self.xs[:, i])
        packinglist.append(self.ys)
        self.rbf = Rbf(*tuple(packinglist), function="multiquadric")

    def f(self, xs):
        """
        Evaluates the surrogate model at xs.
        """
        return self.rbf(*tuple(xs))


class InverseRBF(Surrogate):
    """
    Constructs a surrogate model using the Radial Basis Function in SciPy.
    Uses an Inverse distribution.
    """
    def build(self):
        """
        Builds the surrogate model.
        """
        self.name = "InverseRBF"
        from scipy.interpolate import Rbf

        # creating packing list for passing to RBF
        packinglist = []
        for i in range(len(self.xs[0])):
            packinglist.append(self.xs[:, i])
        packinglist.append(self.ys)
        self.rbf = Rbf(*tuple(packinglist), function="inverse")

    def f(self, xs):
        """
        Evaluates the surrogate model at xs.
        """
        return self.rbf(*tuple(xs))


class LinearRBF(Surrogate):
    """
    Constructs a surrogate model using the Radial Basis Function in SciPy.
    Uses a Linear distribution.
    """
    def build(self):
        """
        Builds the surrogate model.
        """
        self.name = "LinearRBF"
        from scipy.interpolate import Rbf

        # creating packing list for passing to RBF
        packinglist = []
        for i in range(len(self.xs[0])):
            packinglist.append(self.xs[:, i])
        packinglist.append(self.ys)
        self.rbf = Rbf(*tuple(packinglist), function="linear")

    def f(self, xs):
        """
        Evaluates the surrogate model at xs.
        """
        return self.rbf(*tuple(xs))


class CubicRBF(Surrogate):
    """
    Constructs a surrogate model using the Radial Basis Function in SciPy.
    Uses a Cubic distribution.
    """
    def build(self):
        """
        Builds the surrogate model.
        """
        self.name = "CubicRBF"
        from scipy.interpolate import Rbf

        # creating packing list for passing to RBF
        packinglist = []
        for i in range(len(self.xs[0])):
            packinglist.append(self.xs[:, i])
        packinglist.append(self.ys)
        self.rbf = Rbf(*tuple(packinglist), function="cubic")

    def f(self, xs):
        """
        Evaluates the surrogate model at xs.
        """
        return self.rbf(*tuple(xs))


class QuinticRBF(Surrogate):
    """
    Constructs a surrogate model using the Radial Basis Function in SciPy.
    Uses a Quintic distribution.
    """
    def build(self):
        """
        Builds the surrogate model.
        """
        self.name = "QuinticRBF"
        from scipy.interpolate import Rbf

        # creating packing list for passing to RBF
        packinglist = []
        for i in range(len(self.xs[0])):
            packinglist.append(self.xs[:, i])
        packinglist.append(self.ys)
        self.rbf = Rbf(*tuple(packinglist), function="quintic")

    def f(self, xs):
        """
        Evaluates the surrogate model at xs.
        """
        return self.rbf(*tuple(xs))


class ThinPlateRBF(Surrogate):
    """
    Constructs a surrogate model using the Radial Basis Function in SciPy.
    Uses a Thin Plate distribution.
    """
    def build(self):
        """
        Builds the surrogate model.
        """
        self.name = "ThinPlateRBF"
        from scipy.interpolate import Rbf

        # creating packing list for passing to RBF
        packinglist = []
        for i in range(len(self.xs[0])):
            packinglist.append(self.xs[:, i])
        packinglist.append(self.ys)
        self.rbf = Rbf(*tuple(packinglist), function="thin_plate")

    def f(self, xs):
        """
        Evaluates the surrogate model at xs.
        """
        return self.rbf(*tuple(xs))


class BestRBF(Surrogate):
    """
    Constructs a surrogate model based on the RBF with minimal NRMSD
    """
    def calcNRMSD(self, Surrogate):
        """
        Returns the NRMSD of the surrogate model.
        """
        from gc import collect
        predys = np.zeros(len(self.ys))
        for i in range(len(self.ys)):
            inst = Surrogate()
            inst.xs = np.vstack((self.xs[:i], self.xs[i+1:]))
            inst.ys = np.hstack((self.ys[:i], self.ys[i+1:]))
            inst.build()
            predys[i] = inst.f(self.xs[i])
            # explicit dereferencing and cleanup
            del inst
            collect()
        NRMSD = (sum((np.array(predys)-np.array(self.ys))**2)
                 / float(len(self.ys)))**0.5 /\
                (self.maxy() - self.miny())
        # explicit dereferencing
        del predys
        return NRMSD

    def build(self):
        """
        Builds the surrogate model.
        """
        from gc import collect
        # ensure array form for memory efficiency
        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        # define candidate RBFs
        candidates = [GaussianRBF, MultiQuadricRBF, InverseRBF,
                      LinearRBF, CubicRBF, QuinticRBF, ThinPlateRBF]
        # make storage for NRMSD values
        self.NRMSDs = []
        for candidate in candidates:
            # calculate NRMSD for each candidate RBF
            self.NRMSDs.append(self.calcNRMSD(candidate))
            # trigger garbage collection for memory cleanup
            collect()
        # find best RBF
        besti = np.argmin(self.NRMSDs)
        # make best RBF
        brbf = candidates[besti]()
        brbf.xs = self.xs
        brbf.ys = self.ys
        brbf.build()
        # use best RBF as current RBF
        self.name = brbf.name
        self.rbf = brbf.rbf

    def f(self, xs):
        """
        Evaluates the surrogate model at xs.
        """
        return self.rbf(*tuple(xs))


class Kriging(Surrogate):
    """
    A Kriging surrogate model, based on:
    Forrester et al - Engineering Design via Surrogate Modelling (Wiley, 2008).
    including a few optimisation changes to reduce build and evaluation time.
    """
    def __init__(self):
        self.xs = []
        self.ys = []
        self.gapops = 20
        self.gagens = 100
        self.lntlb = -7  # e^-7 ~= 10^-3
        self.lntub = 5   # e^5 ~= 10^2 as per book

    def build(self):
        """
        Builds the surrogate model.
        """
        self.name = "Kriging"
        # extract size parameters
        k = len(self.xs[0])
        n = len(self.ys)
        self.overn = 1/float(n)
        # run ga search of likelihood
        self.Theta, self.MinNegLnLikelihood, _ = \
            ga(lambda x: self.likelihood(x)[0], k,
               np.linspace(self.lntlb, self.lntub, 101),
               self.gapops, self.gagens)
        # put Cholesky factorisation of Psi into namespace
        self.NegLnLike, self.Psi, self.U = self.likelihood(self.Theta)
        return None

    def rebuild(self):
        """
        Rebuilds the surrogate model,
        seeding the optimisation with current Kriging hyper-parameters.
        """
        self.name = "Kriging"
        # extract size parameters
        k = len(self.xs[0])
        n = len(self.ys)
        self.overn = 1/float(n)
        # construct seed
        seedTheta = list(self.Theta)
        # run ga search of likelihood
        self.Theta, self.MinNegLnLikelihood, _ = \
            ga(lambda x: self.likelihood(x)[0], k,
               np.linspace(self.lntlb, self.lntub, 101),
               self.gapops, self.gagens,
               seed=seedTheta)
        # put Cholesky factorisation of Psi into namespace
        self.NegLnLike, self.Psi, self.U = self.likelihood(self.Theta)
        return None

    def likelihood(self, thetas):
        """
        Calculates the likelihood.
        """
        # initialise theta, n, one, eps
        theta = np.e**np.array(thetas)
        n = len(self.ys)
        one = np.ones([n])
        eps = 1000*np.spacing(1)
        # pre-allocate memory
        Psi = np.zeros([n, n])
        # build upper half of the correlation matrix
        for i in range(n):
            for j in range(i+1, n):
                Psi[i, j] = np.exp(-sum(theta*(self.xs[i]-self.xs[j])**2))
        # add upper and lower halves and diagonal of ones
        # plus a small number to reduce ill conditioning
        Psi += Psi.T + np.eye(n) * (1+eps)

        # cholesky factorisation
        # added try/except block to capture error and implement penalty
        try:
            U = npla.cholesky(Psi).T
        except npla.LinAlgError:
            return 1000, Psi, np.zeros([n, n])
        # Forrester et al. have a penalty here if ill-conditioned
        # but this is not implemented in numpy.linalg.cholesky

        # Sum lns of diagonal to find ln(det(Psi))
        LnDetPsi = 2*sum(np.log(np.abs(np.diag(U))))

        # use back-substitution of Cholesky instead of inverse
        mu = np.dot(one, npla.solve(U, npla.solve(U.T, self.ys))) /\
            np.dot(one, npla.solve(U, npla.solve(U.T, one)))
        ysMuTemp = self.ys - mu  # only calculate this once
        SigmaSqr = (np.dot(ysMuTemp,
                           npla.solve(U,
                                      npla.solve(U.T, ysMuTemp)))*self.overn)
        NegLnLike = -1*(-(0.5*n)*np.log(SigmaSqr)-0.5*LnDetPsi)
        return NegLnLike, Psi, U

    def f(self, xs):
        """
        Evaluates the surrogate model at xs.
        """
        # initialise theta
        theta = np.e**np.array(self.Theta)
        # calculate number of sample points
        n = len(self.ys)
        # create vector of ones
        one = np.ones([n])
        # calculate mu
        mu = np.dot(one, npla.solve(self.U, npla.solve(self.U.T, self.ys))) /\
            np.dot(one, npla.solve(self.U, npla.solve(self.U.T, one)))
        psi = np.exp(-np.sum(theta*np.abs(self.xs-xs)**2, axis=1))
        return mu+np.dot(psi,
                         npla.solve(self.U, npla.solve(self.U.T, self.ys-mu)))

    def lb(self, xs):
        """
        Evaluates the statistical lower bound at xs.
        """
        # initialise theta
        theta = np.e**np.array(self.Theta)
        # intialise A
        if not hasattr(self, "A"):
            self.A = 2
        # calculate number of sample points
        n = len(self.ys)
        # create vector of ones
        one = np.ones([n])
        # calculate mu
        mu = np.dot(one, npla.solve(self.U, npla.solve(self.U.T, self.ys))) /\
            np.dot(one, npla.solve(self.U, npla.solve(self.U.T, one)))
        # calculate sigma^2
        ysMuTemp = self.ys - mu  # only calculate this once
        UUTym = npla.solve(self.U, npla.solve(self.U.T, ysMuTemp))
        SigmaSqr = np.dot(ysMuTemp, UUTym)*self.overn

        psi = np.exp(-np.sum(theta*np.abs(self.xs-xs)**2, axis=1))

        # calculate prediction
        f = mu + np.dot(psi, UUTym)
        # error
        SSqr = SigmaSqr*(1-np.dot(psi,
                                  npla.solve(self.U,
                                             npla.solve(self.U.T, psi))))
        # lower bound
        return f - self.A * np.sqrt(SSqr)

    def ei(self, xs):
        """
        Evaluates the expected improvement at xs.
        """
        # define the error function as it's missing from python
        def erf(x):
            # save the sign of x
            sign = 1 if x >= 0 else -1
            x = np.abs(x)

            # constants
            a1 = 0.254829592
            a2 = -0.284496736
            a3 = 1.421413741
            a4 = -1.453152027
            a5 = 1.061405429
            p = 0.3275911

            # A&S formula 7.1.26
            t = (1.0 + p*x)**-1
            y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
            return sign*y  # erf(-x) = -erf(x)
        # initialise theta
        theta = np.e**np.array(self.Theta)
        # intialise A
        if not hasattr(self, "A"):
            self.A = 2
        # calculate number of sample points
        n = len(self.ys)
        # create vector of ones
        one = np.ones([n])
        # calculate mu
        mu = np.dot(one, npla.solve(self.U, npla.solve(self.U.T, self.ys))) /\
            np.dot(one, npla.solve(self.U, npla.solve(self.U.T, one)))
        # calculate sigma^2
        ysMuTemp = self.ys - mu  # only calculate this once
        UUTym = npla.solve(self.U, npla.solve(self.U.T, ysMuTemp))
        SigmaSqr = np.dot(ysMuTemp, UUTym)*self.overn

        psi = np.exp(-np.sum(theta*np.abs(self.xs-xs)**2, axis=1))

        # calculate prediction
        f = mu + np.dot(psi, UUTym)
        y_hat = f
        # error
        SSqr = SigmaSqr*(1-np.dot(psi,
                                  npla.solve(self.U,
                                             npla.solve(self.U.T, psi))))
        # find best so far:
        y_min = np.min(self.ys)
        # expected improvement
        if SSqr == 0:
            return 0
        else:
            sqrtAbsSSqr = np.sqrt(np.abs(SSqr))  # only calculate this once
            yDiff = y_min - y_hat  # only calculate this once
            ei_term1 = yDiff *\
                (0.5+0.5*erf((0.70710678)*(yDiff/sqrtAbsSSqr)))
            ei_term2 = sqrtAbsSSqr *\
                (0.39894228)*np.exp(-0.5*(yDiff**2/SSqr))
            return ei_term1 + ei_term2


# analysis functions
def calcNRMSD(Surrogate, xs, ys):
    """
    Calculates the normalised root mean square deviation of a surrogate class.

    Parameters
    ----------
    Surrogate:  Surrogate
        Surrogate class for which to calculate the NRMSD.
    xs: list of list of numbers
        x-data to calculate NRMSD.
    ys: list of numbers
        y-data to calcualte NRMSD.


    Returns
    NRMSD:  number
        Normalised root mean square deviation.
    """
    from gc import collect
    predys = np.zeros(len(ys))
    deltay = max(ys) - min(ys)
    for i in range(len(ys)):
        inst = Surrogate()
        inst.xs = np.vstack((xs[:i], xs[i+1:]))
        inst.ys = np.hstack((ys[:i], ys[i+1:]))
        inst.build()
        predys[i] = inst.f(xs[i])
        # explicit dereferencing and cleanup
        del inst
        collect()
    NRMSD = (sum((np.array(predys)-np.array(ys))**2)
             / float(len(ys)))**0.5 /\
            (deltay)
    # explicit dereferencing
    del predys
    return NRMSD


# genetic and evolutionary algorithms
def ga(f, length, bases, pops=20, gens=100,
       tournamentSize=0.4, mutationRate=0.6, seed=False):
    """
    A genetic algorithm for finding the minimum of f.

    A genetic algorithm for finding the minimum of f.  Uses a tournament
    selection method and both crossover and mutation to introduce variation.
    Elite selection is also used to preserve the best individual found so far.
    Includes the option to 'seed' the initial population with the placement
    of a predefined individual.

    Parameters
    ----------
    f:  function
        Function to be minimised, that takes a single iterable argument.
    length: integer
        Length of the iterable to pass to the function.
    bases:  iterable
        Possible values for each place in the genetic code.
        For example ["A", "C", "G", "T"] for DNA.
    pops:   number  (default 20)
        Individuals in each new population.
    gens:   number  (default 100)
        Number of generations of populations.
    tournamentSize: number  (default 0.4)
        Size of tournament as a proportion of the population.
    mutationRate:   number  (default 0.6)
        Rate of mutation expressed in the range (0, 1).
    seed:   list of numbers (default False)
        A seed individual to include in the first population.

    Returns
    -------
    indiout:    list of numbers
        The 'chromosome' of the fittest individual found.
    fitness:    number
        The fitness of the output individual.  i.e. f(indiout).
    history:    list of numbers
        The optimisation history, taking the maximum fitness in each generation
        and thus returning a list of length gens.
    """
    import random

    # set up history
    hist = []

    # set up tournament size as integer
    nTournament = int(tournamentSize*pops)

    # generate initial population
    parents = []
    if seed is not False:
        parents.append(seed)
        for i in range(pops-1):
            indi = [random.choice(bases) for j in range(length)]
            parents.append(indi)
    else:
        for i in range(pops):
            indi = [random.choice(bases) for j in range(length)]
            parents.append(indi)

    # calculate fitnesses
    fits = [f(parent) for parent in parents]
    hist.append(min(fits))

    # begin main loop over generations
    for gen in range(gens-1):
        children = []

        # elite selection
        children.append(parents[fits.index(min(fits))])

        # select remaining population
        while len(children) < pops:
            # tournament selection
            tournis = random.sample(range(len(fits)), nTournament)
            # pick the two best parents in the tournament
            p1i = tournis[0]
            p2i = tournis[0]
            for tourni in tournis[1:]:
                if fits[tourni] < fits[p1i]:
                    p1i = tourni
                elif fits[tourni] < fits[p2i]:
                    p2i = tourni

            # crossover
            cp = random.randint(0, length-1)
            child1 = parents[p1i][:cp]+parents[p2i][cp:]
            child2 = parents[p2i][:cp]+parents[p1i][cp:]

            # mutation child1
            if random.random() < mutationRate:
                mp = random.randint(0, length-1)
                child1[mp] = random.choice(bases)
            # mutation child2
            if random.random() < mutationRate:
                mp = random.randint(0, length-1)
                child2[mp] = random.choice(bases)

            # add to population
            children.append(child1)
            if len(children) < pops:
                children.append(child2)

        # progress one generation and recalculate fitness
        parents = children
        fits = [f(parent) for parent in parents]

        # store history
        hist.append(min(fits))

    # find best of final generation
    for i in range(pops):
        if fits[i] == hist[-1]:
            indiout = parents[i]

    return indiout, hist[-1], hist


# sample plan space filling metrics
def sampleplan_mean_distance(sampleplan):
    """
    Returns the mean distance between points in a sample plan.

    Parameters
    ----------
    sampleplan: n*k array
        Sample plan to calculate the mean distance of.

    Returns
    -------
    mean_distance:  number
        Mean distance between points in the sample plan.
    """
    mean_distance = 0
    n = len(sampleplan)
    total_measured = 0
    for i in range(n-1):
        for j in range(i + 1, n):
            mean_distance += npla.norm(sampleplan[i] - sampleplan[j])
            total_measured += 1
    mean_distance /= total_measured
    return mean_distance


def morris_mitchell_phi(sampleplan, q=2, euclidean=True):
    """
    Calculates the sampling plan quality criterion of Morris and Mitchell.

    Parameters
    ----------
    sampleplan: 2d array
        An n by k array of the sample plan.  Where n is the number of points
        and k is the number of dimensions.
    q:  number (default 2)
        Exponent used in the calculation of the metric.
    euclidean:  bool (default True)
        Whether to use the Euclidean distance metric or rectangular.

    Returns
    -------
    phiq:   number
        Sampling plan space-fillingness metric.
    """
    # number of points in sampling plan
    n = len(sampleplan)

    # compute the distances between all pairs of points
    d = np.zeros(n*(n-1)/2.0)
    for i in range(n-1):
        for j in range(i+1, n):
            # d[(i-1)*n-(i-1)*i/2+j-i] is the original matlab here
            if euclidean:
                d[(i)*n-(i)*(i+1)/2+j-i-1] = npla.norm(sampleplan[i] -
                                                       sampleplan[j])
            if not euclidean:
                d[(i)*n-(i)*(i+1)/2+j-i-1] = npla.norm(sampleplan[i] -
                                                       sampleplan[j], 1)

    # remove multiple occurrences
    dd = np.unique(d)

    # preallocate memory for J
    J = np.zeros(len(dd))

    # generate multiplicity array
    for i in range(len(dd)):
        # J[i] = sum(ismember(d, dd[i])) is the original matlab here
        J[i] = sum([x == dd[i] for x in d])

    # the sampling plan quality criterion
    phiq = sum(J*(dd**(-q)))**(1.0/q)

    return phiq


# sampling plans
def randlh(k, n, edges=False):
    """
    Returns an random latin hypercube with k dimensions
    and n points in a structure xs[n][k].
    All dimensions are normalised between 0 and 1.

    Parameters
    ----------
    k:  number
        Number of dimensions.
    n:  number
        Number of points in the latin hypercube.
    edges:  bool (default False)
        Whether or not to use edge points at 0 and 1.

    Returns
    -------
    samplexs:   2d array
        An n by k array of sample points in the given space.

    Example
    -------
    >>> randlh(2, 2)
    [[0.25, 0.25], [0.75, 0.75]]
    """
    from random import randint

    samplexs = np.zeros([n, k])

    # create k by n dimensional sampling list - to be popped at random.
    popper = []
    for i in range(k):
        popper.append(list(range(n)))

    # create latin hypercube
    for i in range(n):
        for j in range(k):
            samplexs[i, j] = popper[j].pop(randint(0, len(popper[j]) - 1))
            # and normalise to 1
            if edges:
                samplexs[i, j] /= float(n - 1)
            elif not edges:
                samplexs[i, j] = (samplexs[i, j] + 0.5) / float(n)

    return samplexs


def bestlh(k, n, n_hypercubes=50, edges=False,
           space_fillingness=morris_mitchell_phi):
    """
    Generates a number of random latin hypercubes
    and picks the best one based on maximum space fillingness.

    Parameters
    ----------
    k:  integer
        Number of dimensions.
    n:  integer
        Number of points in the latin hypercube.
    n_hypercubes:   integer (default 50)
        Number of hypercubes to generate to pick the best one.
    edges:  bool (default False)
        Whether or not to use edge points at 0 and 1.
    space_fillingness:  function    (default morris_mitchell_phi)
        Function that defines the space fillingness of a sample plan.
        This is the objective that is minimised.

    Returns
    -------
    samplexs:   2d array
        An n by k array of sample points in the given space.
    """
    currentxs = randlh(k, n, edges)
    newxs = randlh(k, n, edges)
    if space_fillingness(newxs) < space_fillingness(currentxs):
        currentxs = newxs[:]
    for i in range(n_hypercubes - 2):
        newxs = randlh(k, n, edges)
        if space_fillingness(newxs) < space_fillingness(currentxs):
            currentxs = newxs[:]

    return currentxs


def randsampleplan(k, n):
    """
    Returns a random sample plan.
    All dimensions are normalised between 0 and 1.

    Parameters
    ----------
    k:  integer
        Number of dimensions.
    n:  integer
        Number of points.

    Returns
    -------
    samplexs:   2d array
        An n by k array of sample points in the given space.
    """
    from random import random
    samplexs = np.zeros([n, k])
    for i in range(n):
        for j in range(k):
            samplexs[i, j] = random()
    return samplexs


def bestrandplan(k, n, n_plans=50,
                 space_fillingness=sampleplan_mean_distance):
    """
    Generates a number of random sample plans
    and picks the best one based on maximum space fillingness.

    Parameters
    ----------
    k:  integer
        Number of dimensions.
    n:  integer
        Number of points.
    n_plans:   integer (default 50)
        Number of random plans to generate to pick the best one.
    space_fillingness:  function    (default sampleplan_mean_distance)
        Function that defines the space fillingness of a sample plan.
        This is the objective that is maximised.

    Returns
    -------
    samplexs:   2d array
        An n by k array of sample points in the given space.
    """
    currentxs = randsampleplan(k, n)
    newxs = randsampleplan(k, n)
    if space_fillingness(newxs) > space_fillingness(currentxs):
        currentxs = newxs[:]
    for i in range(n_plans - 2):
        newxs = randsampleplan(k, n)
        if space_fillingness(newxs) > space_fillingness(currentxs):
            currentxs = newxs[:]

    return currentxs


def full2dsampleplan(n):
    """
    Returns a full sample plan for 2 dimensions with n points per dimension.
    Note this is a total of n^2 points.
    All dimensions are normalised between 0 and 1.

    Parameters
    ----------
    n:  integer
        Number of points per dimension.

    Returns
    -------
    samplexs:   2d array
        An n*n by 2 array of sample points in the given space.
    """
    samplexs = np.zeros([n**2, 2])
    values = np.linspace(0, 1, n)
    for i in range(n**2):
        samplexs[i, 0] = values[i//n]
        samplexs[i, 1] = values[i % n]
    return samplexs


def full_factoral_sampleplan(k, n_per_dim):
    """
    Returns a full factoral sample plan for k dimensions
    with n_per_dim points in each dimension.
    N.B. this is a total of n_per_dim**k points.
    All dimensions are normalised between 0 and 1.

    Parameters
    ----------
    k:  number
        Number of dimensions.
    n_per_dim:  number
        Number of points per dimension.

    Returns
    -------
    samplexs:   2d array
        Sample plan of xs.

    Example
    -------
    >>> full_factoral_sampleplan(2, 2)
    [[0, 0], [0, 1], [1, 0], [1, 1]]
    """
    totalxs = n_per_dim**k
    samplexs = np.zeros([totalxs, k])
    current_array = np.zeros(k)
    interval = 1.0/(n_per_dim - 1)
    for i in range(totalxs):
        samplexs[i] = current_array[:]
        for j in range(k):
            if current_array[-(j+1)] + interval <= 1.0:
                current_array[-(j+1)] += interval
                for l in range(j):
                    current_array[-(l+1)] = 0.0
                break
            else:
                pass

    return samplexs


def sobol(k, n):
    """
    Return a sobol sample plan of n points in k dimensions.

    Parameters
    -------
    k:  integer
        Number of dimensions, max 20.
    n:  integer
        Number of points.

    Returns
    -------
    samplexs:   2d array
        Sample plan of xs.
    """
    # N. B. this is directly adapted from sobol.cc by Frances Y. Kuo
    # hence the crappy hacks to use 1-based indexing!
    # Contact: f.kuo@unsw.edu.au

    # bounds check inputs
    if not (1 < k < 21):
        raise(ValueError, "Only 2 to 20 dimensions are supported!")

    # data for generation
    # D = range(21)
    S = [0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7]
    A = [0, 0, 0, 1, 1, 2, 1, 4, 2, 4, 7, 11, 13, 14, 1, 13, 16, 19, 22, 25, 1]
    M_is = [[], [], [1], [1, 3], [1, 3, 1], [1, 1, 1], [1, 1, 3, 3],
            [1, 3, 5, 13], [1, 1, 5, 5, 17], [1, 1, 5, 5, 5],
            [1, 1, 7, 11, 19], [1, 1, 5, 1, 1], [1, 1, 1, 3, 11],
            [1, 3, 5, 5, 31], [1, 3, 3, 9, 7, 49], [1, 1, 1, 15, 21, 21],
            [1, 3, 1, 13, 27, 49], [1, 1, 1, 15, 7, 5], [1, 3, 1, 15, 13, 25],
            [1, 1, 5, 5, 19, 61], [1, 3, 7, 11, 23, 15, 103]]

    samplexs = np.zeros([n, k])
    L = int(np.ceil(np.log(n)/np.log(2.0)))
    C = np.zeros(n, dtype=np.uint32)
    C[0] = 1
    for i in range(1, n):
        C[i] = 1
        value = i
        while value & 1:
            value >>= 1
            C[i] += 1

    V = np.zeros(L+1, dtype=np.uint32)
    for i in range(1, L+1):
        V[i] = 1 << (32 - i)

    X = np.zeros(n, dtype=np.uint32)
    # X[0] = 0
    for i in range(1, n):
        X[i] = X[i-1] ^ V[C[i-1]]
        samplexs[i, 0] = X[i]/float(2**32)

    for j in range(1, k):
        # load in data
        # d = D[j+1]
        s = S[j+1]
        a = A[j+1]
        m = [0]+M_is[j+1]

        V = np.zeros(L+1, dtype=np.uint32)
        if L <= s:
            for i in range(1, L+1):
                V[i] = m[i] << (32 - i)
        else:
            for i in range(1, s+1):
                V[i] = m[i] << (32 - i)
            for i in range(s+1, L+1):
                V[i] = V[i-s] ^ (V[i-s] >> s)
                for k in range(1, s):
                    V[i] ^= (((a >> (s-1-k)) & 1) * V[i-k])

        X = np.zeros(n, dtype=np.uint32)
        # X[0] = 0
        for i in range(1, n):
            X[i] = X[i-1] ^ V[C[i-1]]
            samplexs[i, j] = X[i]/float(2**32)

    return samplexs


def random_binary_spiral(k, n):
    """
    Returns a binary spiral sample plan for n points in k dimensions.

    Parameters
    -------
    k:  integer
        Number of dimensions.
    n:  integer
        Number of points.

    Returns
    -------
    samplexs:   2d array
        Sample plan of xs.
    """
    samplexs = np.zeros([n, k])
    alphas = np.random.choice(np.linspace(0, 1, n*k), size=n*k, replace=False)
    for i in range(n):
        for j in range(k):
            samplexs[i, j] = 0.5 + np.random.choice([-1, 1])*0.5*alphas[j+i*k]
    return samplexs


# test functions and other utilities
def p_norm(x1, x2=False, p=2):
    """
    Returns the p norm of vector x1 or optionally, the vector x1->x2
    in n-dimensional space.

    Parameters
    ----------
    x1: list
        First (or only) vector.
    x2: list or False   (default False)
        Optional second vector.
    p:  integer (default 2)
        The p-value of the norm to take.

    Returns
    -------
    norm:   number
        Value of the p_norm.
    """
    if np.array(x2).any():
        norm = sum([(x - y)**p for x, y in zip(x2, x1)])**(1.0/p)
    else:
        norm = sum([x**p for x in x1])**(1.0/p)
    return norm


def normdims(k):
    """
    Returns a list of tuples to give limits for k normalised dimensions.
    i.e. [(0,1)]*k

    Parameters
    ----------
    k:  integer
        Number of dimensions.

    Returns
    -------
    dims:   2d array
        An 2 by k array of normalised bounds i.e. [(0,1)]*k.
    """
    return [(0, 1)]*k


def onevar(xs):
    """
    Single variable test function.

    from Forrester, Sobester, Keane
    - Engineering Design via Surrogate Modelling.
    Minimum at: (0.75725)
    Value of:   -6.0207

    Parameters
    ----------
    xs: 1d array
        Value of x.

    Returns
    -------
    y:  number
        (6*x-2)**2 * np.sin(12*x-4).
    """
    x = xs[0]
    return (6*x-2)**2 * np.sin(12*x-4)


def branin(xs):
    """
    Two variable test function.

    from Forrester, Sobester, Keane
    - Engineering Design via Surrogate Modelling.
    Normalised to [0,1] bounds.
    Minimum at: (0.08736553,  0.90889357)
    Value of:   -16.644021

    Parameters
    ----------
    xs: length 2 1d array
        Value of x.

    Returns
    -------
    y:  number
        Value of the branin function at x.
    """
    # convert 0,1 limits to x1<-[-5,10],x2<-[0,15]
    x1 = 15*xs[0]-5
    x2 = 15*xs[1]

    return (x2 - (5.1*x2)/(4*np.pi*np.pi) + (5*x1)/(np.pi) - 6)**2 +\
        10*((1 - 1/(8*np.pi))*np.cos(x1)+1) + 5*x1


def rastrigin(xs):
    """
    Return the value of the Rastrigin function with implicit dimensions.

    Minimum at: (0)^k
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Values of Rastrigin function at xs.
    """
    d = len(xs)
    y = 10*d
    for i in range(d):
        y += xs[i]**2 - 10*np.cos(2*np.pi*xs[i])
    return y


def rastrigin_norm(xs):
    """
    Return the value of the Rastrigin function normalised to [0,1] bounds.

    Minimum at: (0.5)^k
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point between [0,1]^d

    Returns
    -------
    y:  number
        Values of Rastrigin function at -5.12 + (xs*10.24).
    """
    nxs = [-5.12 + x*10.24 for x in xs]
    return rastrigin(nxs)


def ackleys(xs):
    """
    Return the value of Ackley's function with two dimensions.

    Minimum at: (0, 0)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Ackley's function at xs.
    """
    x1 = xs[0]
    x2 = xs[1]
    y = -20*np.exp(-0.2 * np.sqrt(0.5*(x1**2 + x2**2)))
    y += -np.exp(0.5*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))) + 20 + np.e
    return y


def ackleys_norm(xs):
    """
    Return the value of Ackley's function with two normalised dimensions.

    Minimum at: (0.5, 0.5)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Ackley's function at the normalised xs.
    """
    return ackleys([(xs[0]-0.5)*10, (xs[1]-0.5)*10])


def sphere(xs):
    """
    Return the value of the Sphere function with multiple dimensions.

    Minimum at: (0)^k
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of the Sphere function at xs.
    """
    y = 0
    for x in xs:
        y += x**2
    return y


def rosenbrock(xs):
    """
    Return the value of the Rosenbrock function with multiple dimensions.

    Minimum at: (1)^k
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of the Rosenbrock function at xs.
    """
    y = 0
    for i in range(len(xs)-1):
        y += 100 * (xs[i+1]-xs[i]**2)**2 + (xs[i]-1)**2
    return y


def beales(xs):
    """
    Return the value of Beale's function with two dimensions.

    Minimum at: (3, 0.5)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Beale's function at xs.
    """
    x1 = xs[0]
    x2 = xs[1]
    y = (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2*x2)**2 +\
        (2.625 - x1 + x1*x2*x2*x2)**2
    return y


def beales_norm(xs):
    """
    Return the value of Beale's function with two normalised dimensions.

    Minimum at: (0.8333, 0.5556)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Beale's function at normalised xs.
    """
    return beales([xs[0]*9-4.5, xs[1]*9-4.5])


def goldstein_price(xs):
    """
    Return the value of the Goldstein-Price function with two dimensions.

    Minimum at: (0, -1)
    Value of:   3

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of the Goldstein-Price function at xs.
    """
    x1 = xs[0]
    x2 = xs[1]
    y = (1 + (x1 + x2 + 1)**2 *
         (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)) *\
        (30 + (2*x1 - 3*x2)**2 *
         (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
    return y


def goldstein_price_norm(xs):
    """
    Return the value of the Goldstein-Price function with normalised dims.

    Minimum at: (0.5, 0.25)
    Value of:   3

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of the Goldstein-Price function at normalised xs.
    """
    return goldstein_price([xs[0]*4-2, xs[1]*4-2])


def booths(xs):
    """
    Return the value of Booth's function with two dimensions.

    Minimum at: (1, 3)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Booth's function at xs.
    """
    return (xs[0] + 2*xs[1] - 7)**2 + (2*xs[0] + xs[1] - 5)**2


def booths_norm(xs):
    """
    Return the value of Booth's function with two normalised dimensions.

    Minimum at: (0.55, 0.65)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Booth's function at normalised xs.
    """
    return booths([xs[0]*20-10, xs[1]*20-10])


def bukin6(xs):
    """
    Return the value of Bukin function #6 with two dimensions.

    Minimum at: (-10, 1)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Bukin function #6 at xs.
    """
    return 100 * np.sqrt(np.abs(xs[1]-0.01*xs[0]**2)) + 0.01*np.abs(xs[0]+10)


def bukin6_norm(xs):
    """
    Return the value of Bukin function #6 with two normalised dimensions.

    Minimum at: (0.5, 0.6667)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Bukin function #6 at normalised xs.
    """
    return bukin6([xs[0]*10-15, xs[1]*6-3])


def matyas(xs):
    """
    Return the value of Matyas function with two dimensions.

    Minimum at: (0, 0)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Matyas function at xs.
    """
    return 0.26*(xs[0]**2 + xs[1]**2) - 0.48*xs[0]*xs[1]


def matyas_norm(xs):
    """
    Return the value of Matyas function with two normalised dimensions.

    Minimum at: (0.5, 0.5)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Matyas function at normalised xs.
    """
    return matyas([xs[0]*20-10, xs[1]*20-10])


def levi(xs):
    """
    Return the value of Levi function with two dimensions.

    Minimum at: (1, 1)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Levi function at xs.
    """
    y = np.sin(3*np.pi*xs[0])**2
    y += (xs[0] - 1)**2 * (1 + np.sin(3*np.pi*xs[1])**2)
    y += (xs[1] - 1)**2 * (1 + np.sin(2*np.pi*xs[1])**2)
    return y


def levi_norm(xs):
    """
    Return the value of Levi function with two normalised dimensions.

    Minimum at: (0.55, 0.55)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Levi function at normalised xs.
    """
    return levi([xs[0]*20-10, xs[1]*20-10])


def three_hump_camel(xs):
    """
    Return the value of Three-hump Camel function with two dimensions.

    Minimum at: (0, 0)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Three-hump Camel function at xs.
    """
    return 2*xs[0]**2 - 1.05*xs[0]**4 + xs[0]**6/6.0 + xs[0]*xs[1] + xs[1]**2


def three_hump_camel_norm(xs):
    """
    Return the value of Three-hump Camel function with two normalised dims.

    Minimum at: (0.5, 0.5)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Three-hump Camel function at normalised xs.
    """
    return three_hump_camel([xs[0]*10-5, xs[1]*10-5])


def easom(xs):
    """
    Return the value of Easom function with two dimensions.

    Minimum at: (pi, pi)
    Value of:   -1

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Easom function at xs.
    """
    return -np.cos(xs[0])*np.cos(xs[1]) *\
        np.exp(-((xs[0]-np.pi)**2+(xs[1]-np.pi)**2))


def easom_norm(xs):
    """
    Return the value of Easom function with two normalised dimensions.

    Minimum at: (0.5157, 0.5157)
    Value of:   -1

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Easom function at xs.
    """
    return easom([xs[0]*200-100, xs[1]*200-100])


def cross_in_tray(xs):
    """
    Return the value of Cross-In-Tray function with two dimensions.

    Minimum at: (+-1.34941, +-1.34941)
    Value of:   -2.06261

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Cross-In-Tray function at xs.
    """
    return -0.0001*(np.abs(np.sin(xs[0])*np.sin(xs[1]) *
                    np.exp(np.abs(100 -
                                  np.sqrt(xs[0]**2+xs[1]**2)/np.pi)))+1)**0.1


def cross_in_tray_norm(xs):
    """
    Return the value of Cross-In-Tray function with two normalised dimensions.

    Minimum at: (0.5+-0.0674705, 0.5+-0.0674705)
    Value of:   -2.06261

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Cross-In-Tray function at normalised xs.
    """
    return cross_in_tray([xs[0]*20-10, xs[1]*20-10])


def eggholder(xs):
    """
    Return the value of Eggholder function with two dimensions.

    Minimum at: (512, 404.2319)
    Value of:   -959.6407

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Eggholder function at xs.
    """
    return - (xs[1]+47)*np.sin(np.sqrt(np.abs(xs[1]+0.5*xs[0]+47))) -\
        xs[0]*np.sin(np.sqrt(np.abs(xs[0]-xs[1]-47)))


def eggholder_norm(xs):
    """
    Return the value of Eggholder function with two normalised dimensions.

    Minimum at: (1, 0.89476)
    Value of:   -959.6407

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Eggholder function at normalised xs.
    """
    return eggholder([xs[0]*1024-512, xs[1]*1024-512])


def holder_table(xs):
    """
    Return the value of Holder table function with two dimensions.

    Minimum at: (+-8.05502, +-9.66459)
    Value of:   -19.2085

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Holder table function at xs.
    """
    return -np.abs(np.sin(xs[0])*np.cos(xs[1]) *
                   np.exp(np.abs(1-np.sqrt(xs[0]**2+xs[1]**2)/np.pi)))


def holder_table_norm(xs):
    """
    Return the value of Holder table function with two normalised dimensions.

    Minimum at: (0.5+-0.402751, 0.5+-0.49832295)
    Value of:   -19.2085

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Holder table function at normalised xs.
    """
    return holder_table([xs[0]*20-10, xs[1]*20-10])


def mccormick(xs):
    """
    Return the value of McCormick function with two dimensions.

    Minimum at: (-0.54719, -1.54719)
    Value of:   -1.9133

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of McCormick function at xs.
    """
    return np.sin(xs[0]+xs[1]) + (xs[0]-xs[1])**2 - 1.5*xs[0] + 2.5*xs[1] + 1


def mccormick_norm(xs):
    """
    Return the value of McCormick function with two normalised dimensions.

    Minimum at: (0.17324, 0.20754)
    Value of:   -1.9133

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of McCormick function at normalised xs.
    """
    return mccormick([xs[0]*5.5-1.5, xs[1]*7-3])


def schaffer2(xs):
    """
    Return the value of Schaffer #2 function with two dimensions.

    Minimum at: (0, 0)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Schaffer #2 function at xs.
    """
    return 0.5 + (np.sin(xs[0]**2-xs[1]**2)**2 - 0.5) /\
        ((1 + 0.001*(xs[0]**2+xs[1]**2))**2)


def schaffer2_norm(xs):
    """
    Return the value of Schaffer #2 function with two normalised dimensions.

    Minimum at: (0.5, 0.5)
    Value of:   0

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Schaffer #2 function at normalised xs.
    """
    return schaffer2([xs[0]*200-100, xs[1]*200-100])


def schaffer4(xs):
    """
    Return the value of Schaffer #4 function with two dimensions.

    Minimum at: (0, 1.25313)
    Value of:   0.292579

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Schaffer #4 function at xs.
    """
    return 0.5 + (np.cos(np.sin(np.abs(xs[0]**2-xs[1]**2)))**2 - 0.5) /\
        ((1 + 0.001*(xs[0]**2+xs[1]**2))**2)


def schaffer4_norm(xs):
    """
    Return the value of Schaffer #4 function with two normalised dimensions.

    Minimum at: (0.5, 0.50626565)
    Value of:   0.292579

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Schaffer #4 function at normalised xs.
    """
    return schaffer4([xs[0]*200-100, xs[1]*200-100])


def styblinski_tang(xs):
    """
    Return the value of Styblinksi-Tang function with multiple dimensions.

    Minimum at: (-2.903534)^k
    Value of:   -39.16599*k

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Styblinksi-Tang function at xs.
    """
    return 0.5*(sum([x**4 - 16*x**2 + 5*x for x in xs]))


def styblinski_tang_norm(xs):
    """
    Return the value of Styblinksi-Tang function with multiple normalised dims.

    Minimum at: (0.2096466)^k
    Value of:   -39.16599*k

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    y:  number
        Value of Styblinksi-Tang function at normalised xs.
    """
    return styblinski_tang(10*np.array(xs)-5)


def fonseca_fleming(xs):
    """
    Return the values of Fonseca and Fleming functions.

    Functions:  2
    Dimensions: k
    Bounds:     -4 to 4

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Fonseca and Fleming functions at xs.
    """
    f1s = 0
    f2s = 0
    k = len(xs)
    for x in xs:
        f1s += (x - 1/np.sqrt(k))**2
        f1s += (x + 1/np.sqrt(k))**2
    f1 = 1 - np.exp(-f1s)
    f2 = 1 - np.exp(-f2s)
    return f1, f2


def fonseca_fleming_norm(xs):
    """
    Return the values of normalised Fonseca and Fleming functions.

    Functions:  2
    Dimensions: k
    Bounds:     0 to 1

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Fonseca and Fleming functions at normalised xs.
    """
    return fonseca_fleming(np.array(xs)*8-4)


def kursawe(xs):
    """
    Return the values of Kursawe functions.

    Functions:  2
    Dimensions: 3
    Bounds:     -5 to 5

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Kursawe functions at xs.
    """
    f1 = 0
    f2 = 0
    for i in range(2):
        f1 += -10*np.exp(-0.2*np.sqrt(xs[i]**2+xs[i+1]**2))
        f2 += np.abs(xs[i])**0.8 + 5*np.sin(xs[i]**3)
    f2 += np.abs(xs[2])**0.8 + 5*np.sin(xs[2]**3)
    return f1, f2


def kursawe_norm(xs):
    """
    Return the values of normalised Kursawe functions.

    Functions:  2
    Dimensions: 3
    Bounds:     0 to 1

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Kursawe functions at normalised xs.
    """
    return kursawe(np.array(xs)*10-5)


def schafferM1(xs):
    """
    Return the values of Schaffer #1 functions.

    Functions:  2
    Dimensions: 1
    Bounds:     -10 to 10

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Schaffer #1 functions at xs.
    """
    return xs[0]**2, (xs[0]-2)**2


def schafferM1_norm(xs):
    """
    Return the values of normalised Schaffer #1 functions.

    Functions:  2
    Dimensions: 1
    Bounds:     0 to 1

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Schaffer #1 functions at xs.
    """
    return schafferM1([xs[0]*20-10])


def schafferM2(xs):
    """
    Return the values of Schaffer #2 functions.

    Functions:  2
    Dimensions: 1
    Bounds:     -5 to 10

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Schaffer #2 functions at xs.
    """
    x = xs[0]
    if x <= 1:
        f1 = -x
    elif x <= 3:
        f1 = x-2
    elif x <= 4:
        f1 = 4-x
    else:
        f1 = x-4
    f2 = (x-5)**2
    return f1, f2


def schafferM2_norm(xs):
    """
    Return the values of normalised Schaffer #2 functions.

    Functions:  2
    Dimensions: 1
    Bounds:     0 to 1

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Schaffer #2 functions at normalised xs.
    """
    return schafferM2([xs[0]*15-5])


def poloni(xs):
    """
    Return the values of Poloni functions.

    Functions:  2
    Dimensions: 2
    Bounds:     -pi to pi

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Poloni functions at xs.
    """
    A1 = 0.5*np.sin(1) - 2*np.cos(1) + np.sin(2) - 1.5*np.cos(2)
    A2 = 1.5*np.sin(1) - np.cos(1) + 2*np.sin(2) - 0.5*np.cos(2)
    B1 = lambda x, y: 0.5*np.sin(x) - 2*np.cos(x) + np.sin(y) - 1.5*np.cos(y)
    B2 = lambda x, y: 1.5*np.sin(x) - np.cos(x) + 2*np.sin(y) - 0.5*np.cos(y)
    f1 = 1 + (A1 - B1(xs[0], xs[1]))**2 + (A2 - B2(xs[0], xs[1]))**2
    f2 = (xs[0] + 3)**2 + (xs[1] + 1)**2
    return f1, f2


def poloni_norm(xs):
    """
    Return the values of normalised Poloni functions.

    Functions:  2
    Dimensions: 2
    Bounds:     -pi to pi

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Poloni functions at normalised xs.
    """
    return poloni(np.array(xs)*(2*np.pi)-np.pi)


def ZDT1(xs):
    """
    Return the values of Zitzler-Deb-Thiele #1 functions.

    Functions:  2
    Dimensions: 1 to 30
    Bounds:     0 to 1

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Zitzler-Deb-Thiele #1 functions at xs.
    """
    f1 = xs[0]
    gx = 1.0 + (9/29.0)*sum(xs[1:])
    f2 = gx*(1 - np.sqrt(f1/gx))
    return f1, f2


def ZDT2(xs):
    """
    Return the values of Zitzler-Deb-Thiele #2 functions.

    Functions:  2
    Dimensions: 1 to 30
    Bounds:     0 to 1

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Zitzler-Deb-Thiele #2 functions at xs.
    """
    f1 = xs[0]
    gx = 1.0 + (9/29.0)*sum(xs[1:])
    f2 = gx*(1 - (f1/gx)**2)
    return f1, f2


def ZDT3(xs):
    """
    Return the values of Zitzler-Deb-Thiele #3 functions.

    Functions:  2
    Dimensions: 1 to 30
    Bounds:     0 to 1

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Zitzler-Deb-Thiele #3 functions at xs.
    """
    f1 = xs[0]
    gx = 1.0 + (9/29.0)*sum(xs[1:])
    f2 = gx*(1 - np.sqrt(f1/gx) - (f1/gx)*np.sin(10*np.pi*f1))
    return f1, f2


def ZDT4(xs):
    """
    Return the values of Zitzler-Deb-Thiele #4 functions.

    Functions:  2
    Dimensions: 2 to 10
    Bounds:     0 < x1 < 1, -5 < xi < 5

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Zitzler-Deb-Thiele #4 functions at xs.
    """
    f1 = xs[0]
    gx = 91.0
    for x in xs[1:]:
        gx += x**2 - 10*np.cos(4*np.pi*x)
    f2 = gx*(1 - np.sqrt(f1/gx))
    return f1, f2


def ZDT4_norm(xs):
    """
    Return the values of normalised Zitzler-Deb-Thiele #4 functions.

    Functions:  2
    Dimensions: 2 to 10
    Bounds:     0 to 1

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Zitzler-Deb-Thiele #4 functions at normalised xs.
    """
    return ZDT4([xs[0]]+[x*10-5 for x in xs[1:]])


def ZDT6(xs):
    """
    Return the values of Zitzler-Deb-Thiele #6 functions.

    Functions:  2
    Dimensions: 1 to 10
    Bounds:     0 to 1

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Zitzler-Deb-Thiele #6 functions at xs.
    """
    f1 = 1 - np.exp(-4*xs[0])*np.sin(6*np.pi*xs[0])**6
    gx = 1.0 + 9*(sum(xs[1:])/9.0)**0.25
    f2 = gx*(1 - (f1/gx)**2)
    return f1, f2


def viennet(xs):
    """
    Return the values of Viennet functions.

    Functions:  3
    Dimensions: 2
    Bounds:     -3 to 3

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Viennet functions at xs.
    """
    f1 = 0.5*(xs[0]**2+xs[1]**2)+np.sin(xs[0]**2+xs[1]**2)
    f2 = 0.125*(3*xs[0]-2*xs[1]+4)**2 + (xs[0]-xs[1]+1)**2/27.0 + 15
    f3 = 1.0/(xs[0]**2+xs[1]**2+1) - 1.1*np.exp(-(xs[0]**2+xs[1]**2))
    return f1, f2, f3


def viennet_norm(xs):
    """
    Return the values of normalised Viennet functions.

    Functions:  3
    Dimensions: 2
    Bounds:     -3 to 3

    Parameters
    ----------
    xs: list or array
        Co-ordinates of x point.

    Returns
    -------
    ys: tuple of numbers
        Values of Viennet functions at normalised xs.
    """
    return viennet(np.array(xs)*6-3)
