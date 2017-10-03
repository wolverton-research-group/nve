import sys
from collections import defaultdict
import re
import scipy.spatial
import numpy as np
import pulp


# regex for formula
re_formula = re.compile('([A-Z][a-z]?)([0-9\.]*)')


def is_integer(number, tol=1e-5):
    number = float(number)
    return abs(round(number) - number) < tol


def parse_comp(value):
    comp = defaultdict(float)
    for elt, amt in re_formula.findall(value):
        if elt in ['D', 'T']:
            elt = 'H'
        if amt == '':
            comp[elt] = 1
        elif is_integer(amt):
            comp[elt] += int(round(float(amt)))
        else:
            comp[elt] += float(amt)
    return dict(comp)


def normalize_dict(dictionary):
    tot = float(sum(dictionary.values()))
    return dict((k, v/tot) for k,v in dictionary.items())


def unit_comp(comp):
    return normalize_dict(comp)


def parse_space(value):
    if isinstance(value, basestring):
        space = re.sub('[-,_]', ' ', value)
        space = [ unit_comp(parse_comp(b)) for b in space.split()]
    elif isinstance(value, (list,set)):
        space = [ {elt:1} for elt in value ]
    elif isinstance(value, dict):
        space = [ {elt:1} for elt in value ]
    elif not value:
        space = None
    else:
        raise ValueError("Failed to parse space: %s" % value)
    return space


def get_phase_coord(comp_str, space):
    comp = parse_comp(comp_str)
    comp = defaultdict(float, comp)
    bounds = parse_space(space)
    bound_space = set.union(*[set(b.keys()) for b in bounds])
    basis = []
    for b in bounds:
        basis.append([b.get(k, 0) for k in sorted(bound_space)])
    basis = np.array(basis)
    bcomp = dict((k, v) for k, v in comp.items() if k in bound_space)
    comp = unit_comp(bcomp)
    cvec = np.array([comp.get(k, 0) for k in sorted(bound_space)])
    coord = np.linalg.lstsq(basis.T, cvec)[0]
    return list(coord)


def convex_hull_phases(phases, axes='CEV'):
    """Calculate the phases that lie on the composition-volume-pressure convex hull of a given list of phases.

    :param phases: dictionary of thermodynamic data to be used to construct the convex hull
                   e.g., {'Al2Si_13125': {'icsd_id': 1654, 'comp_str': 'Al2Si', 'energy_pa': -1.5, ...}, ...}
    :type phases: dict(str, dict(str, str or int or float))
    :param axes: composition(C)/energy(E)/volume(V) or a combination of them
                 i.e., one of 'CE', 'CEV', or 'CV' ('EV' is meaningless, FYI)
    :type axes: str
    :return: dictionary of phases on the composition-volume-pressure convex hull. See :param phases: above.
    :rtype: dict(str, dict(str, str or int or float)); See :type phases: above.
    """
    # A = list of phase coordinates
    # e.g., Al0.66Si0.33 with energy = -1.5 eV/atom and volume = 15.0 Ang^3/atom is appended to A as [0.33, -1.5, 15.0]
    A = []
    for phase_dict in phases:
        if axes == 'CE':
            A.append(phase_dict['phase_coord'][1:] + [phase_dict['energy_pa']])
        elif axes == 'CV':
            A.append(phase_dict['phase_coord'][1:] + [phase_dict['volume_pa']])
        elif axes == 'CEV':
            A.append(phase_dict['phase_coord'][1:] + [phase_dict['energy_pa'], phase_dict['volume_pa']])
        else:
            raise NotImplementedError

    # use `scipy.spatial.ConvexHull` to construct the convex hull
    chull = scipy.spatial.ConvexHull(A)

    hull = []
    for facet in chull.simplices:
        if any([ind >= len(phases) for ind in facet]):
            continue
        face = [phases[ind] for ind in facet if ind < len(phases)]
        hull.append(face)

    hull_phases = {}
    for face in hull:
        for phase in face:
            k = '{}_{}'.format(phase['comp_str'], phase['oqmd_id'])
            hull_phases[k] = phase
    return hull_phases


def gclp_energy(gclp_phases, hull_phases):
    return sum([hull_phases[p]['energy_pa']*xp for p, xp in gclp_phases.items()])


def pressure_gclp(composition, phases=None, volume=None, verbosity='low'):
    """Perform minimization of energy at a given composition and volume.

    :param composition: total composition at which energy should be minimized
    :type composition: str
    :param phases: dictionary of phases on the composition-volume-pressure convex hull
                   e.g., {'Al2Si_13125': {'icsd_id': 1654, 'comp_str': 'Al2Si', 'energy_pa': -1.5, ...}, ...}
    :type phases: dict(str, dict(str, str or int or float))
    :param volume: volume per atom at which energy should be minimized. Defaults to None.
    :type volume: float
    :param verbosity: standard output verbosity; one of low/l, medium/m, high/h. Defaults to 'low'.
                      M/h ouput warnings. Errors are always printed.
    :type verbosity: str
    :return: dictionary of ground state phases and phase fractions
    :rtype: dict(str, float)
    """

    ucomp = unit_comp(parse_comp(composition))
    if not phases:
        sys.stdout.write('Please input phase data at function call!\n')
        sys.stdout.flush()
        sys.exit(1)

    elems = set()  # elems = python set of elements at the given composition
    for comp_oqmd_id, phase_dict in phases.items():
        for e in phase_dict['unit_comp'].keys():
            elems.add(e)

    comp_vec = {}  # composition vector
    for e in elems:
        comp_vec[e] = ucomp.get(e, 0)

    if verbosity.lower() in ['high', 'h', 'medium', 'med', 'm']:
        sys.stdout.write('Given composition: {}, '.format(composition))
        sys.stdout.write('Composition vector: {}\n'.format(str(comp_vec)))
        sys.stdout.flush()

    if volume is None:
        prob = pulp.LpProblem('min{E=f(comp)}', pulp.LpMinimize)
    else:
        prob = pulp.LpProblem('min{E=f(comp, volume)}', pulp.LpMinimize)
    # define names for the variables (compositions)
    x = pulp.LpVariable.dicts('x', phases.keys(), 0., 1.)
    # objective function
    # Sum_{all p} x_p * E_p
    prob += pulp.lpSum([x[k]*v['energy_pa'] for k, v in phases.items()]), 'Energy'
    # Composition constraint
    # Sum_{all p} x_p = X_0 (input composition)
    for e, c in comp_vec.items():
        prob += pulp.lpSum([x[k]*v['unit_comp'].get(e, 0) for k, v in phases.items()]) == float(c),\
                'Conservation of {}'.format(e)
    # Volume constraint
    # Sum_{all p} x_p * v_p = V_0 (input composition)
    if volume is not None:
        prob += pulp.lpSum([x[k]*v['volume_pa'] for k, v in phases.items()]) == float(volume), 'Conservation of volume'

    if verbosity.lower() in ['high', 'h']:
        sys.stdout.write('LP Problem: {}\n'.format(str(prob)))

    # Solve the LP problem
    prob.solve()

    gclp_phases = dict([(p, x[p].varValue) for p in phases if x[p].varValue > 1e-6])
    if verbosity.lower() in ['high', 'h']:
        sys.stdout.write('LP Problem: {}\n'.format(str(prob)))

    if verbosity.lower() in ['high', 'h', 'medium', 'med', 'm']:
        sys.stdout.write('Ground state phases: ')
        gclp_phases_str = ', '.join(['{}: {}'.format(p, xp) for p, xp in gclp_phases.items()])
        sys.stdout.write('{}\n'.format(gclp_phases_str))
        sys.stdout.write('Ground state energy (hull energy): {:0.4f}\n'.format(gclp_energy(gclp_phases)))

    return gclp_phases


def sum_phase_comp(phases):
    sum_phases = 0.
    for p, xp in phases.items():
        sum_phases += xp
    return sum_phases


def pressure_stability_range(phase, all_phases, verbosity='low'):
    """
    :param phase: thermodynamic phase data for a phase (see `all_phases` below)
                  e.g., {'oqmd_id': 1654, 'comp_str': 'Al2Si', 'energy_pa': -1.5, ...}
    :type phase: dict
    :param all_phases: dictionary of thermodynamic data to be used to construct the convex hull
                       e.g., {'Al2Si_13125': {'icsd_id': 1654, 'comp_str': 'Al2Si', 'energy_pa': -1.5, ...}, ...}
    :type all_phases: dict(str, dict(str, str or int or float))
    :param verbosity: standard output verbosity; one of low/l, medium/m, high/h. Defaults to 'low'.
                      M/h ouput warnings. Errors are always printed.
    :type verbosity: str
    :return: dictionary of range of pressures in which various phases are predicted to be stable.
    :rtype: dict(str, dict(str, str or int or float))
    """
    phase_key = '_'.join([phase['comp_str'], str(phase['oqmd_id'])])

    stability_range_data = {}
    stability_range_data[phase_key] = phase
    stability_range_data[phase_key].update({'errors': [],
                                            'comments': [],
                                            'on_CEV_hull': False,
                                            'min_pressure': None,
                                            'max_pressure': None})

    evang3_to_gpa = 160.21766208
    dv = 1E-4
    elems = phase['unit_comp'].keys()
    space = '-'.join(sorted(elems))

    phases = []
    for _phase in all_phases:
        if set(_phase['unit_comp'].keys()).issubset(set(elems)):
            _phase.update({'phase_coord': get_phase_coord(_phase['comp_str'], space)})
            phases.append(_phase)
    if verbosity.lower() in ['high', 'h', 'medium', 'med', 'm']:
        sys.stdout.write('Number of phases in space {} = {}\n'.format(space, len(phases)))
    if not phases:
        stability_range_data[phase_key]['errors'].append(['Not enough phases in the chemical space.'])
        return stability_range_data[phase_key]

    # get the phases on the convex hull
    try:
        hull_phases = convex_hull_phases(phases)
    except scipy.spatial.qhull.QhullError:
        stability_range_data[phase_key]['errors'].append(['Not enough phases to constuct simplex.'])
        return stability_range_data[phase_key]

    # check if the phase of interest is on the CEV convex hull
    if phase_key in hull_phases:
        stability_range_data[phase_key].update({'on_CEV_hull': True})
    else:
        return stability_range_data[phase_key]

    vhull_phases = {}
    if len(elems) > 1:
        # get the phases on the volume-composition convex hull
        vhull_phases = convex_hull_phases(phases, axes='CV')

    phase_volumes = [v['volume_pa'] for k, v in hull_phases.items()]
    for k, v in hull_phases.items():
        # do this only for the phase of interest
        if k != phase_key:
            continue
        phase_energy = v['energy_pa']
        # eliminate the spurious phases
        gclp_phases = pressure_gclp(v['comp_str'], phases=hull_phases, volume=v['volume_pa'])
        hull_energy = gclp_energy(gclp_phases, hull_phases)
        if abs(phase_energy-hull_energy) > 1E-6:
            stability_range_data[k].update({'on_CEV_hull': False})
            comment = 'E(phase on the hull) != E(hull at same composition).'
            stability_range_data[k]['comments'].append(comment)
            if verbosity.lower() in ['high', 'h', 'medium', 'med', 'm']:
                sys.stdout.write('[C] {} '.format(comment))
                sys.stdout.write('{}: E(phase) = {:0.4f}, E(hull) = {:0.4f}.\n'.format(k, phase_energy, hull_energy))
            return stability_range_data[phase_key]

        # perform LP at V+dV and V-dV to get the pressure stability windows
        gclp_phases_plus = pressure_gclp(v['comp_str'], phases=hull_phases, volume=v['volume_pa']+dv)
        gclp_phases_minus = pressure_gclp(v['comp_str'], phases=hull_phases, volume=v['volume_pa']-dv)

        # The pressure potential is given simply by the numerical derivative (E(V_0) - E(V_0 +/- dV))/dV
        min_pressure = (phase_energy-gclp_energy(gclp_phases_plus, hull_phases))/dv*evang3_to_gpa
        max_pressure = (gclp_energy(gclp_phases_minus, hull_phases)-phase_energy)/dv*evang3_to_gpa

        # if the phase has the highest/lowest volume of all the phases, the lowest/highest stable pressure = -/+ inf
        if abs(v['volume_pa'] - min(phase_volumes)) <= 1e-5:
            max_pressure = float('inf')
        if abs(v['volume_pa'] - max(phase_volumes)) <= 1e-5:
            min_pressure = float('-inf')

        # check for the boundaries of the phase space
        if k in vhull_phases:
            phase_comp_plus = sum_phase_comp(gclp_phases_plus)
            phase_comp_minus = sum_phase_comp(gclp_phases_minus)
            p_by_m = phase_comp_plus/phase_comp_minus
            m_by_p = phase_comp_minus/phase_comp_plus
            comment = 'Phase on the C-V hull. '
            comment += '{}: C+ = {:0.6f}, C- = {:0.6f}, '.format(k, phase_comp_plus, phase_comp_minus)
            comment += 'C+/C- = {:0.6f}, C-/C+ = {:0.6f}'.format(p_by_m, m_by_p)
            stability_range_data[k]['comments'].append(comment)
            if verbosity.lower() in ['high', 'h', 'medium', 'med', 'm']:
                sys.stdout.write('[W] {}\n'.format(comment))
            if abs(p_by_m) > 1.000001:
                min_pressure = float('-inf')
            if abs(m_by_p) > 1.000001:
                max_pressure = float('inf')

        # finally, if min_pressure is greater than max_pressure, something is definitely bonkers.
        if min_pressure > max_pressure:
            error = 'P_min > P_max!'
            stability_range_data[k]['errors'].append(error)
            sys.stdout.write('[E] {} '.format(error))
            sys.stdout.write('{}: P_min = {:0.3f}, P_max = {:0.3f}\n'.format(k, min_pressure, max_pressure))
            return stability_range_data[phase_key]

        stability_range_data[k].update({'min_pressure': min_pressure, 'max_pressure': max_pressure})
    return stability_range_data[phase_key]

