from math import *
import json
import sys
import numpy as np
import pprint
import os
import pickle
import random
from scipy.stats import norm
from scipy.stats import skewnorm


class Options:
    """
    Stores all needed options, read from a config.json
    """

    def __init__(self, config_dict):
        """
        Initializes all attributes of the optionhandler from a config_dict
        Args:
            config_dict (dict): dict that holds all values for the option parameters
        """

        self.vl = config_dict['vl']  # [mL] Volume of lagoons
        self.tl = config_dict['tl']  # [min] nTime in lagoons
        self.vt = config_dict['vt']  # [mL] transferred volume
        self.ceu0 = config_dict['ceu0']  # [cfu/ml] starting concentration of E. coli in new lagoon
        self.cp0 = config_dict['cp0']  # [pfu/ml] concentration of phage that is transferred to the very first lagoon
        self.k = config_dict['k']  # binding constant of phage to E. coli
        self.tu = config_dict['tu']  # [min] doubling time of uninfected E. coli
        self.ti = config_dict['ti']  # [min] doubling time of infected E. coli
        self.tp = config_dict['tp']  # [min] doubling time of phage producing E. coli
        self.mumax = config_dict['mumax']  # [pfu/min] maximum production rate of phage from infected E. coli
        self.tpp = config_dict['tpp']  # [min] time until phage production starts
        self.f0 = config_dict['f0']  #  fitness of phage
        self.max_cp = config_dict['max_cp']  # [cfu] maximum phage titer for meta data evaluation
        self.min_cp = config_dict['min_cp']  # [cfu] minimum phage titer for meta data evaluation
        self.fend = config_dict['fend']  # maximum fitness
        # Values needed for calculation:
        self.tsteps = config_dict['tsteps']  # [min] length of timestep
        self.epochs = config_dict['epochs']  # an epoch is the time between two transfers
        self.capacity = config_dict['capacity']  # [cfu/ml] maximum concentration E.coli can reach under the conditions
        self.dt = self.tl / self.tsteps

        self.growth_mode = config_dict['growth_mode']  # exp or logistic
        self.phageonly = config_dict['phageonly']  # Bool that decides, if initial concentrations
        self.noisy = config_dict['noisy']  # float that scales the noise added to parameters at each call
        self.fitnessmode = config_dict['fitnessmode'] # either lin or const or linear, dist
        self.f_prec = config_dict['f_prec']  # number of different f-values that are possible
        self.to_mutate = config_dict['to_mutate']  # share of share of fitness that is mutated
        self.sigma = config_dict['sigma']  # sigma for gaussian mutation in mutation
        self.mutation_dist = config_dict['mutation_dist']  # distribution for mutatino, either norm or skew
        self.skewness = config_dict['skewness']  # parameter of skewed normal distribution
        self.plot_dist = config_dict['plot_dist']  # wether or not to plot concentrations for each fitness
        try:
            self.phage_deg = config_dict['phage_degradation']  # Percentage of phage that dies per timestep
        except:
            self.phage_deg = 0

class Values:
    """
    Stores all values of a setup, espcially the titers, their derivatives and the time
    """

    def __init__(self, o):
        """Initializes the values object depending on an Options object
        Args:
            o(Options): holds the options for which the Values object is built
        """

        self.f = o.f0
        self.t = o.dt
        self.time = [0]
        self.t_curr_lagoon = 0
        self.current_epoch = 0
        self.epoch = 0

        self.ceu = [o.ceu0]  # [cfu] concentration of uninfected E. coli
        self.cei = [0]  # [cfu] concentration of infected E. coli
        self.cep = [0]  # [cfu] concentration of productive E. coli
        self.cp = [o.cp0]  # [pfu] concentration of phage

        # save all the derivatives
        self.sdceu = [0]
        self.sdcei = [0]
        self.sdcep = [0]
        self.sdcp = [0]
        self.ts = 0

        self.dist_f = [{}]

        first = True
        for i in range(o.f_prec):
            f = i / (o.f_prec - 1)
            if f > o.f0 and first:
                self.dist_f[0][(i - 1) / (o.f_prec - 1)] = 1.0
                self.dist_f[0][f] = 0.0
                first = False
            else:
                self.dist_f[0][f] = 0.0



def initialisation(config_dict):
    """
    Initialises both a Values and an Options object
    Args:
        config_dict(dict): config dict to be passed to the Options init
    Returns:
        o(Options): Object that holds all options
        v(Values): Object that is used to store Concentrations, their derivatives, the time.
    """

    o = Options(config_dict)
    v = Values(o)
    return o, v

# functional dependencies:


def e_growth_rate(current_concentration, td, o, v):
    """
    Calculates growth rate of E.coli
    Args:
        current_concentration(float): current concentration of E. coli
        td (float): Duration between two steps
        o(Options): Options object for lookup
        v(Values): Values object for lookup
    Returns (float): growth rate for the growth mode specified in o.
    """

    if o.growth_mode == 'exp':
        return (log(2) / td) * current_concentration
    elif o.growth_mode == 'logistic':
        return (log(2)/td) * current_concentration * ((o.capacity - (v.cep[-1] + v.cei[-1] + v.ceu[-1]))/o.capacity)
    else:
        raise ValueError('Dont know that growthmode: {}'.format(o.growth_mode))


def mu(current_concentration,  o, v):
    """
    For flexibiliy, not implemented yet
    Args:
        current_concentration(float): current concentration of M13 phage
     o(Options): Options object for lookup
        v(Values): Values object for lookup
    Returns (float): production rate of phage, depending on o
    """
    return o.mumax


def current_f(o, v):
    """
    Args:
        o(Options): Options object for lookup
        v(Values): Values object for lookup
    Returns (float): current fitness, if fitness is not defined distributional
    """

    if o.fitnessmode == 'lin':
        return ((o.fend - o.f0) / (o.epochs-1)) * v.current_epoch + o.f0
    elif o.fitnessmode == 'const':
        return o.f0
    else:
        print('Dont know {} fitness.'.format(o.fitnessmode))


def d(value):
    """
    Discretizes variables that are > 0, sets them to zero if < 1
    Args:
        alue(float): value to be discretized
    Returns (float): discretized value
    """
    if value < 1:
        return 0
    else:
        return value


def g(value, o, sigma=1):
    """
    Adds gaussian noise to a given value, depending on the local argument sigma and the global o.noisy
    Args:
        value(float): Value to be noised
        o(Options): Options object for lookup
        sigma(float): Sigma for normal distribution, if smaller the noise is concentratec to a smaller range (optional).
    Returns(float): The input value with gaussian noise as specified by sigma and o.noisy
    """
    noisy = float(np.random.normal(value, abs(sigma*o.noisy*value)))
    return noisy


def dceu(ts, o, v):
    """
    [cfu/min] change of concentration of uninfected E. coli
    Args:
        ts(int): current time step
        o(Options): Options object for lookup
        v(Values): Values object for lookup
    Returns(float): Derivative of ceu at ts, the change in the concentration of uninfected E. coli between two timesteps
    """
    ceu = g(v.ceu[ts - 1], o)
    cp = g(v.cp[ts - 1], o)
    tu = g(o.tu, o)
    k = g(o.k, o)

    return e_growth_rate(ceu, tu, o, v) - k * ceu * cp


def dcei(ts, o, v):
    """
        [cfu/min] change of concentration of infected E. coli.
        Args:
            ts(int): current time step
            o(Options): Options object for lookup
            v(Values): Values object for lookup
        Returns(float): Derivative of cei at ts, the change in the concentration of infected E. coli between two timesteps
    """

    cei = g(v.cei[ts - 1], o)
    ceu = g(v.ceu[ts - 1], o)
    cp  = g(v.cp[ts - 1], o)
    ti  = g(o.ti, o)
    k   = g(o.k, o)
    tpp = g(o.tpp, o)
    if v.t_curr_lagoon > tpp:
        try:
            sdcei = g(v.sdcei[ts - int(o.tpp / o.dt) - 1], o)
        except:
            sdcei = g(v.sdcei[0], o) #for noisy tpp
        return e_growth_rate(cei, ti, o, v) + k * ceu * cp - sdcei
    else:
        return e_growth_rate(cei, ti, o, v) + k * ceu * cp


def dcep(ts, o, v):
    """
        [cfu/min] change of concentration of productive E. coli
        Args:
            ts(int): current time step
            o(Options): Options object for lookup
            v(Values): Values object for lookup
        Returns(float): Derivative of cep at ts, the change in the concentration of phage-producing E. coli
        between two timesteps
    """

    cep = g(v.cep[ts - 1], o)
    tp = g(o.tp, o)
    tpp = g(o.tpp, o)

    if v.t_curr_lagoon > tpp:
        try:
            sdcei = g(v.sdcei[ts - int(o.tpp / o.dt) - 1], o)
        except:
            sdcei = g(v.sdcei[0], o) # for noisy tpp
        return e_growth_rate(cep, tp, o, v) + max(0, sdcei)
    else:
        return e_growth_rate(cep, tp, o, v)


def dcp(ts, o, v):
    """
        [pfu/min] change of concentration of phage, only for non-distributional fitness
         Args:
             ts(int): current time step
             o(Options): Options object for lookup
             v(Values): Values object for lookup
         Returns(float): Derivative of cp at ts, the change in the concentration of phage between two timesteps
     """

    cep = g(v.cep[ts - 1], o)
    ceu = g(v.ceu[ts - 1], o)
    cp = g(v.cp[ts - 1], o)
    f = g(current_f(o, v), o)
    k = g(o.k, o)
    return cep * mu(cp, o, v) * f - k * ceu * cp


def setup(o, v, logfile):
    """
    Calculates predcel for one set of parameters specified by o, writes everything into v.
    Only for non-distributional fitness
    Args:
        o(Options): Options object for lookup
        v(Values): Values object for lookup and writing
    logfile(open writable file): Current information is logged in this file
    """

    logfile.write('Calculating setup with f0 = {}, vt = {}, tl = {}\n'.format(o.f0, o.vt, o.tl))
    logfile.flush()
    logfile.write('t; Phage; E. coli\n')
    for epoch in range(o.epochs):
        for _ in range(int(o.tsteps)):
            v.epoch += 1
            v.t += o.dt
            v.t_curr_lagoon += o.dt
            v.ts += 1
            v.time.append(v.t)
            v.sdceu.append(dceu(v.ts, o, v))
            v.sdcei.append(dcei(v.ts,  o, v))
            v.sdcep.append(dcep(v.ts, o, v))
            v.sdcp.append(dcp(v.ts, o, v))
            # Euler this one!
            v.ceu.append(d(v.ceu[v.ts - 1] + v.sdceu[v.ts] * o.dt))
            v.cei.append(d(v.cei[v.ts - 1] + v.sdcei[v.ts] * o.dt))
            v.cep.append(d(v.cep[v.ts - 1] + v.sdcep[v.ts] * o.dt))
            v.cp.append(d((1-o.phage_deg)*(v.cp[v.ts - 1] + v.sdcp[v.ts] * o.dt)))
            logfile.write('{}; {}; {}\n'.format(v.time[-1], v.cp[-1], (v.ceu[-1]+v.cei[-1]+v.cep[-1])))

        print('Epoch {} of {}, {}%.'.format(epoch, o.epochs, int(100 * epoch/o.epochs)))

        if o.phageonly == 'True':
            v.t_curr_lagoon = 0
            v.ts += 1
            v.ceu.append(o.ceu0)
            v.cei.append(0)
            v.cep.append(0)
            v.cp.append(transfer(v.cp[-1], o))

            v.sdceu.append(0)
            v.sdcei.append(0)
            v.sdcep.append(0)
            v.sdcp.append(0)
            v.time.append(v.t)

        else:
            v.t_curr_lagoon = 0
            v.ts += 1
            v.ceu.append(transfer(v.ceu[-1], o) + o.ceu0)
            v.cei.append(transfer(v.cei[-1], o))
            v.cep.append(transfer(v.cep[-1], o))
            v.cp.append(transfer(v.cp[-1], o))

            v.sdceu.append(0)
            v.sdcei.append(0)
            v.sdcep.append(0)
            v.sdcp.append(0)
            v.time.append(v.t)

        v.current_epoch += 1

def transfer(cin, o):
    """
    Calculates the dilution of a concentration that happens by one transfer
    Args:
        cin(float): concentration that is transfered
        o(Options): Options object for lookup
    Returns(float): cin normalized by the relation of the transfer volume to the lagoon volume
    """
    return d(cin * o.vt / o.vl)


def meta(config_dict, delta, num_datapoints, logfile):
    """
    Calculates the different sets of options, builds Options and Values objects for them and calculates the whole setup
    for every combination. Provides a nested dict with the obtained Values Objects.
    Args:
        config_dict (dict): A config dict that is the basis for the dicts passed to initialisation to build the Options objects
        delta(float): The relative maximum increase or maximum decrease of f0, tl, vl
        um_datapoints(int): Defines how many values of f0, tl and vl are evaluated. CAUTION metas complexity scales with
        num_datapoints^3!
        logfile(open writable file): Passed to setup(), current information is logged in this file
    Returns(dict):
        Dict that is nested three times, levels are: f0, tl, vl. Stores Values objects for every combination
    """

    count = 0
    data = {}
    start_config_dict = config_dict.copy()
    for fr in range(num_datapoints):  # f0
        f0 = start_config_dict['f0'] * (1 - delta) + fr * start_config_dict['f0'] * (2 * delta / (num_datapoints - 1))
        config_dict['f0'] = f0
        data[f0] = {}

        for tr in range(num_datapoints):  # tl
            tl = start_config_dict['tl'] * (1 - delta) + tr * start_config_dict['tl'] * (2 * delta / (num_datapoints - 1))
            config_dict['tl'] = tl
            data[f0][tl] = {}

            for vr in range(num_datapoints):  # vt
                vt = start_config_dict['vt'] * (1 - delta) + vr * start_config_dict['vt'] * (2 * delta / (num_datapoints - 1))
                config_dict['vt'] = vt

                o, v = initialisation(config_dict)
                if o.fitnessmode == 'dist':
                    dist_setup(o, v, logfile)
                else:
                    setup(o, v, logfile)

                data[f0][tl][vt] = v

                count +=1
                print('{} % ready.'.format(count/(num_datapoints*num_datapoints*num_datapoints)))

    return data


def evaluate_meta(data, o):
    """
    Evaluates the data calculated by meta()
    Args:
        data(dict): nested data dict calculated by meta
        o (Options): Options object for lookup
    Returns(dict): A dictionaray nested in the same way as data, with the three levels f0, tl, vl that contains dicts
    with keys
        cpend(float): final phage titer
        valid_epochs(int): number of epochs in which the phage titer was between mincp and maxcp as defined in Options
        object o.
        tendency(int):
            -1 if the phage titer was lower than mincp at some transfer point
            0 if the phage titer was valid the whole time
            1 if the phage titer was higher than maxcp at some transfer point
    """

    reduced = {}  # get datas structure
    for key1 in list(data.keys()):
        reduced[key1] = {}
        for key2 in list(data[key1].keys()):
            reduced[key1][key2] = {}
            for key3 in list(data[key1][key2].keys()):

                reduced[key1][key2][key3] = {}  # overwrite data still left from data
                reduced[key1][key2][key3]['cpend'] = data[key1][key2][key3].cp[-2]
                valid_epochs, tendency = valid_phage_titer(data[key1][key2][key3].cp[:-1], o)
                reduced[key1][key2][key3]['valid_epochs'] = valid_epochs
                reduced[key1][key2][key3]['tendency'] = tendency
    return reduced

def valid_phage_titer(titer, o):
    """
    Decides wether or not a phage titer is valid and, if not decides why.
    Args:
        titer(float): Titer to decide on.
        o (Options): Options object for lookup
    Returns(int):
            -1 if the phage titer was lower than mincp at some transfer point
            0 if the phage titer was valid the whole time
            1 if the phage titer was higher than maxcp at some transfer point
    """

    gen = 1

    while o.min_cp < titer[(gen * (o.tsteps + 1)) - 1] < o.max_cp:
        if gen == o.epochs:
            return gen, 0
        gen += 1

    if o.min_cp > titer[(gen * (o.tsteps + 1)) - 1]:
        tendency = -1
    else:
        tendency = 1

    return gen, tendency


def calc(config_json, destination, num_datapoints, logfile):
    """
    Controls the calculation, interface to what happens outside this script
    Args:
        config_json(str): path to a config.json file that specifies all needed Options
        destination(str): path to write data and logs into
        num_datapoints(int): Number of datapoints that meta() evaluates
    Returns the full data dict from meta() as well as the reduced data dict from evaluate_meta()
    """

    delta = 0.75
    logfile.write('delta: {}'.format(delta))
    print(destination)
    if not os.path.exists(destination):
        os.mkdir(destination)

    with open(config_json) as config_fobj:
        config_dict = json.load(config_fobj)
    o, v = initialisation(config_dict)

    logfile.write('Calculating data.\n')
    data = meta(config_dict, delta, num_datapoints, logfile)

    logfile.write('Calculating metadata.\n')
    reduced = evaluate_meta(data, o)

    logfile.write('Done.')
    return reduced, data


def dist_setup(o, v, logfile):
    """
    Calculates predcel for one set of parameters specified by o, writes everything into v.
    Only for distributional fitness.
    Args:
        o(Options): Options object for lookup
        v(Values): Values object for lookup and writing
        logfile(open writable file): Current information is logged in this file
    """

    logfile.write('Calculating setup with f0 = {}, vt = {}, tl = {}\n'.format(o.f0, o.vt, o.tl))
    logfile.flush()

    for epoch in range(o.epochs):
        for _ in range(int(o.tsteps)):
            v.epoch += 1
            v.t += o.dt
            v.t_curr_lagoon += o.dt
            v.ts += 1
            v.time.append(v.t)
            v.sdceu.append(dceu(v.ts, o, v))
            v.sdcei.append(dcei(v.ts,  o, v))
            v.sdcep.append(dcep(v.ts, o, v))
            v.sdcp.append(dist_dcp(v.ts, o, v))
            # Euler this one!
            v.ceu.append(d(v.ceu[v.ts - 1] + v.sdceu[v.ts] * o.dt))
            v.cei.append(d(v.cei[v.ts - 1] + v.sdcei[v.ts] * o.dt))
            v.cep.append(d(v.cep[v.ts - 1] + v.sdcep[v.ts] * o.dt))
            v.cp.append(d((1-o.phage_deg) * (v.cp[v.ts - 1] + v.sdcp[v.ts] * o.dt)))

        print('Epoch {} of {}, {}%.'.format(epoch, o.epochs, int(100 * epoch/o.epochs)))

        if o.phageonly == 'True':
            v.t_curr_lagoon = 0
            v.ts += 1
            v.ceu.append(o.ceu0)
            v.cei.append(0)
            v.cep.append(0)
            v.cp.append(transfer(v.cp[-1], o))

            v.sdceu.append(0)
            v.sdcei.append(0)
            v.sdcep.append(0)
            v.sdcp.append(0)
            v.time.append(v.t)
            v.dist_f.append(v.dist_f[-1])

        else:
            v.t_curr_lagoon = 0
            v.ts += 1
            v.ceu.append(transfer(v.ceu[-1], o) + o.ceu0)
            v.cei.append(transfer(v.cei[-1], o))
            v.cep.append(transfer(v.cep[-1], o))
            v.cp.append(transfer(v.cp[-1], o))

            v.sdceu.append(0)
            v.sdcei.append(0)
            v.sdcep.append(0)
            v.sdcp.append(0)
            v.time.append(v.t)
            v.dist_f.append(v.dist_f[-1])

        v.current_epoch += 1

def dist_dcp(ts, o, v):
    """
        [pfu/min] change of concentration of phage, only for distributional fitness
         Args:
             ts(int): current time step
             o(Options): Options object for lookup
             v(Values): Values object for lookup
         Returns(float): Derivative of cp at ts, the change in the concentration of phage between two timesteps
     """

    cep = v.cep[ts - 1]
    ceu = v.ceu[ts - 1]
    cp = v.cp[ts - 1]
    k = o.k

    # output
    f = 0

    cache_f = {}

    dist_f = v.dist_f[max(0, ts - int(o.tpp / o.dt) - 1)]

    for f_val in list(dist_f.keys()):
        f_share  = dist_f[f_val]
        # produced phage with this fitness

        dcp_f = f_share * f_val

        f += dcp_f
        cache_f[f_val] = dcp_f

    current_f_dist = {}
    total = np.sum(list(cache_f.values()))
    for key in list(cache_f.keys()):
        current_f_dist[key] = cache_f[key]/total

    current_f_dist = mutation(current_f_dist, o) # mutation(v.dist_f[-1], o) to disable selection

    v.dist_f.append(current_f_dist)

    return f * mu(cp, o, v) * g(cep, o) - k * ceu * cp


def mutation(fin, o):
    """
    Simulates Mutation by applying noise to the fitness distribution. THe noise is either from a normal distribution or
    from a skewed normal distribution as defined in o.
    Args:
        fin(dict): Dict that stores the distribution: fitness values as keys and their share from the total concentration as values,
        mutation is simulated
        o(Options): Options object for lookup
    Returns(dict):
        Dict that stores the distribution: fitness values as keys and their share from the total concentration as values,
        differs from fin by the added  noise
    """
    fout = {}
    width = 1 / (o.f_prec - 1)
    for f_val in list(fin.keys()):
        fout[f_val] = (1 - o.to_mutate) * fin[f_val]
    for f_val in list(fin.keys()):

        gf_val = g(f_val, o)
        gf_share  = g(fin[f_val], o)
        skew_sum = 0
        for fo_val in list(fin.keys()):
            # upper part:
            if o.mutation_dist == 'norm':
                f_increase = (
                    norm.cdf(
                        x=fo_val + 0.5 * width if fo_val != 1.0 else float('inf'),
                        loc=f_val,
                        scale=o.sigma
                ) - norm.cdf(
                        x=fo_val - 0.5 * width if fo_val != 0.0 else - float('inf'),
                        loc=f_val,
                        scale=o.sigma
                )) * gf_share * o.to_mutate

            elif o.mutation_dist == 'skew':
                f_increase = (
                 skewnorm.cdf(
                     x=fo_val + 0.5 * width if fo_val != 1.0 else float('inf'),
                     a=o.skewness,
                     loc=f_val,
                     scale=o.sigma
                 ) - skewnorm.cdf(
                     x=fo_val - 0.5 * width if fo_val != 0.0 else - float('inf'),
                     a=o.skewness,
                     loc=f_val,
                     scale=o.sigma
                )) * gf_share * o.to_mutate

            else:
                print('Unknown mutation_dist {}'.format(o.mutation_dist))
                return
            fout[fo_val] += max(0, f_increase) # to minimize floating point errors


        total = np.sum(list(fout.values()))
        for key in list(fout.keys()):
            fout[key] = fout[key] / total

    return fout
