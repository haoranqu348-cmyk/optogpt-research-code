"""
joint_sp/self_improving/perturb.py — GA/PSO perturbation for joint s+p structures.

Genetic Algorithm:
  - Individual fitness: joint s+p TMM error
  - Material mutation: only from ALLOWED_MATERIALS (NO conductors!)
  - Thickness mutation: quantized to checkpoint thickness tokens

PSO:
  - Particle position = continuous thickness vector
  - Quantize to nearest valid thickness token after optimization
  - Fitness: joint s+p error

Joint error: 0.5 * error(sim_s, target_s) + 0.5 * error(sim_p, target_p)
For high-T targets: extra weight on worst_pol mean_T
"""

import copy
import hashlib
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_SP = _SCRIPT_DIR.parent
_PKG_ROOT = _JOINT_SP.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from joint_sp.constants import (
    ALLOWED_MATERIALS, BANNED_MATERIALS,
    THETA_DEG, SUBSTRATE, SUBSTRATE_THICK_NM, MAX_LAYERS, SPEC_DIM, BRANCH_DIM,
    MIN_IMPROVEMENT,
)

DEFAULT_WL = np.arange(0.4, 1.101, 0.01)


def structure_hash_from_tokens(tokens):
    return hashlib.sha256("|".join(tokens).encode()).hexdigest()


# ============================================================
# TMM Fitness
# ============================================================

def _compute_fitness(materials, thicknesses, target_joint, nk_dict,
                     theta=THETA_DEG, substrate=SUBSTRATE,
                     substrate_thick=SUBSTRATE_THICK_NM):
    """
    Compute joint s+p fitness of a structure.

    Args:
        materials: list of material names
        thicknesses: list of thickness values (nm)
        target_joint: (284,) target [Rs, Ts, Rp, Tp]
        nk_dict: material refractive index dict

    Returns:
        dict with E_s, E_p, E_joint, mean_Ts, mean_Tp, etc.
    """
    from optogpt.core.datasets.sim import spectrum

    target_Rs = target_joint[0:71]
    target_Ts = target_joint[71:142]
    target_Rp = target_joint[142:213]
    target_Tp = target_joint[213:284]

    try:
        # s-pol
        sim_s = spectrum(materials, thicknesses, pol='s', theta=theta,
                          wavelengths=DEFAULT_WL, nk_dict=nk_dict,
                          substrate=substrate, substrate_thick=substrate_thick)
        n_pts = len(sim_s) // 2
        sim_Rs = np.array(sim_s[:n_pts])
        sim_Ts = np.array(sim_s[n_pts:])

        # p-pol
        sim_p = spectrum(materials, thicknesses, pol='p', theta=theta,
                          wavelengths=DEFAULT_WL, nk_dict=nk_dict,
                          substrate=substrate, substrate_thick=substrate_thick)
        sim_Rp = np.array(sim_p[:n_pts])
        sim_Tp = np.array(sim_p[n_pts:])

        # Interpolate if needed
        if n_pts != 71:
            from scipy.interpolate import interp1d
            wl_target = np.linspace(0.4, 1.1, 71)
            wl_orig = np.linspace(0.4, 1.1, n_pts)
            sim_Rs = interp1d(wl_orig, sim_Rs, kind='linear', fill_value='extrapolate')(wl_target)
            sim_Ts = interp1d(wl_orig, sim_Ts, kind='linear', fill_value='extrapolate')(wl_target)
            sim_Rp = interp1d(wl_orig, sim_Rp, kind='linear', fill_value='extrapolate')(wl_target)
            sim_Tp = interp1d(wl_orig, sim_Tp, kind='linear', fill_value='extrapolate')(wl_target)

    except Exception as e:
        return {'E_joint': 1e10, 'error': str(e)}

    E_s = float(np.mean(np.abs(sim_Rs - target_Rs) + np.abs(sim_Ts - target_Ts)) / 2)
    E_p = float(np.mean(np.abs(sim_Rp - target_Rp) + np.abs(sim_Tp - target_Tp)) / 2)
    E_joint = 0.5 * E_s + 0.5 * E_p
    
    mae_Rs = float(np.mean(np.abs(sim_Rs - target_Rs)))
    mae_Ts = float(np.mean(np.abs(sim_Ts - target_Ts)))
    mae_Rp = float(np.mean(np.abs(sim_Rp - target_Rp)))
    mae_Tp = float(np.mean(np.abs(sim_Tp - target_Tp)))

    mean_Ts = float(np.mean(sim_Ts))
    mean_Tp = float(np.mean(sim_Tp))
    p05_Ts = float(np.percentile(sim_Ts, 5))
    p05_Tp = float(np.percentile(sim_Tp, 5))
    min_Ts = float(np.min(sim_Ts))
    min_Tp = float(np.min(sim_Tp))
    
    # High-T loss
    high_T_loss = float(
        0.35 * np.mean(1.0 - sim_Ts) + 0.35 * np.mean(1.0 - sim_Tp) +
        0.15 * np.percentile(1.0 - sim_Ts, 95) + 0.15 * np.percentile(1.0 - sim_Tp, 95) +
        0.10 * (1.0 - min(mean_Ts, mean_Tp)) +
        0.05 * (1.0 - min(min_Ts, min_Tp))
    )

    return {
        'E_s': E_s, 'E_p': E_p, 'E_joint': E_joint,
        'mae_Rs': mae_Rs, 'mae_Ts': mae_Ts, 'mae_Rp': mae_Rp, 'mae_Tp': mae_Tp,
        'mean_Ts': mean_Ts, 'mean_Tp': mean_Tp,
        'p05_Ts': p05_Ts, 'p05_Tp': p05_Tp,
        'min_Ts': min_Ts, 'min_Tp': min_Tp,
        'worst_pol_mean_T': min(mean_Ts, mean_Tp),
        'high_T_loss': high_T_loss,
        'sim_Rs': sim_Rs, 'sim_Ts': sim_Ts,
        'sim_Rp': sim_Rp, 'sim_Tp': sim_Tp,
    }


def _objective_value(details, objective):
    if objective == 'joint_error':
        return float(details.get('E_joint', 1e10))
    if objective == 'high_transmission':
        return float(details.get('E_joint', 1e10) + details.get('high_T_loss', 1e10))
    raise ValueError(f"Unknown optimization objective: {objective}")


# ============================================================
# Genetic Algorithm
# ============================================================

class Individual:
    """GA individual representing a multilayer structure."""
    def __init__(self, materials, thicknesses):
        self.materials = list(materials)
        self.thicknesses = list(thicknesses)
        self.fitness = None
        self.fitness_details = None

    def copy(self):
        ind = Individual(copy.deepcopy(self.materials), copy.deepcopy(self.thicknesses))
        ind.fitness = self.fitness
        if self.fitness_details is not None:
            ind.fitness_details = {
                k: (v.copy() if hasattr(v, 'copy') else copy.deepcopy(v))
                for k, v in self.fitness_details.items()
            }
        return ind


def _ga_mutate_material(individual, allowed_materials, rng):
    """Replace one random material with another allowed material (NOT banned!)."""
    if len(individual.materials) == 0:
        return
    idx = rng.randint(0, len(individual.materials))
    new_mat = allowed_materials[rng.randint(0, len(allowed_materials))]
    individual.materials[idx] = new_mat


def _ga_mutate_thickness(individual, valid_thicknesses, rng):
    """Perturb one thickness to a nearby valid value."""
    if len(individual.thicknesses) == 0:
        return
    idx = rng.randint(0, len(individual.thicknesses))
    current = individual.thicknesses[idx]

    # Find nearest valid thicknesses
    diffs = np.abs(np.array(valid_thicknesses) - current)
    nearest_idx = np.argmin(diffs)
    # Choose within ±3 steps
    delta = rng.randint(-3, 4)
    new_idx = np.clip(nearest_idx + delta, 0, len(valid_thicknesses) - 1)
    individual.thicknesses[idx] = valid_thicknesses[new_idx]


def _ga_crossover(parent1, parent2, rng):
    """Single-point crossover of materials and thicknesses."""
    min_len = min(len(parent1.materials), len(parent2.materials))
    if min_len < 2:
        return parent1.copy()

    point = rng.randint(1, min_len)
    child_mats = parent1.materials[:point] + parent2.materials[point:]
    child_thicks = parent1.thicknesses[:point] + parent2.thicknesses[point:]

    # Ensure valid length
    if len(child_mats) > MAX_LAYERS:
        child_mats = child_mats[:MAX_LAYERS]
        child_thicks = child_thicks[:MAX_LAYERS]

    return Individual(child_mats, child_thicks)


def ga_optimize(initial_structure, target_joint, nk_dict,
                population_size=50, generations=30, mutation_rate=0.3,
                allowed_materials=None, valid_thicknesses=None,
                rng=None, theta=THETA_DEG, objective='joint_error'):
    """
    GA optimization for joint s+p structure.

    Args:
        initial_structure: dict with 'materials' and 'thicknesses'
        target_joint: (284,) target spectrum
        nk_dict: material nk
        population_size, generations, mutation_rate: GA hyperparams
        allowed_materials: list of allowed material names
        valid_thicknesses: list of valid thickness values (from vocab)
        rng: numpy RandomState

    Returns:
        dict with best structure, fitness, and history
    """
    if rng is None:
        rng = np.random.RandomState()

    if allowed_materials is None:
        allowed_materials = ALLOWED_MATERIALS
    if valid_thicknesses is None:
        valid_thicknesses = list(range(10, 201, 10))

    # Evaluate fitness function
    def evaluate(ind):
        fitness = _compute_fitness(ind.materials, ind.thicknesses, target_joint, nk_dict, theta=theta)
        ind.fitness = _objective_value(fitness, objective)
        ind.fitness_details = fitness
        return fitness

    # Initialize population
    base = Individual(initial_structure['materials'], initial_structure['thicknesses'])
    population = [base.copy()]

    for _ in range(population_size - 1):
        ind = base.copy()
        if rng.random() < 0.5:
            _ga_mutate_material(ind, allowed_materials, rng)
        _ga_mutate_thickness(ind, valid_thicknesses, rng)
        population.append(ind)

    best = None
    best_fitness = float('inf')
    history = []

    for gen in range(generations):
        # Evaluate
        for ind in population:
            if ind.fitness is None:
                evaluate(ind)
            if ind.fitness < best_fitness:
                best_fitness = ind.fitness
                best = ind.copy()

        history.append({
            'generation': gen,
            'best_fitness': best_fitness,
            'pop_avg': np.mean([ind.fitness for ind in population if ind.fitness is not None]),
        })

        # Selection (tournament)
        new_pop = []
        # Elitism: keep best
        new_pop.append(best.copy())

        while len(new_pop) < population_size:
            # Tournament selection
            t_size = 3
            tournament = [population[rng.randint(0, len(population))] for _ in range(t_size)]
            p1 = min(tournament, key=lambda x: x.fitness if x.fitness is not None else float('inf'))

            tournament = [population[rng.randint(0, len(population))] for _ in range(t_size)]
            p2 = min(tournament, key=lambda x: x.fitness if x.fitness is not None else float('inf'))

            # Crossover
            child = _ga_crossover(p1, p2, rng)

            # Mutation
            if rng.random() < mutation_rate:
                _ga_mutate_material(child, allowed_materials, rng)
            if rng.random() < mutation_rate:
                _ga_mutate_thickness(child, valid_thicknesses, rng)

            new_pop.append(child)

        population = new_pop

    # Ensure best has valid fitness_details (re-evaluate if needed)
    if best is not None and best.fitness_details is None:
        best.fitness_details = evaluate(best)
    
    return {
        'materials': best.materials if best else [],
        'thicknesses': best.thicknesses if best else [],
        'fitness': best_fitness,
        'details': best.fitness_details if best else {},
        'history': history,
    }


# ============================================================
# PSO (simplified)
# ============================================================

def pso_optimize(initial_structure, target_joint, nk_dict,
                 swarm_size=30, iterations=50, w=0.7, c1=1.5, c2=1.5,
                 valid_thicknesses=None, rng=None, theta=THETA_DEG,
                 objective='joint_error'):
    """
    Simplified PSO for thickness optimization (materials fixed).

    Since PSO operates on continuous space, thicknesses are optimized
    continuously and then quantized to the nearest valid thickness.

    Args:
        initial_structure: dict with 'materials' and 'thicknesses'
        target_joint: (284,) target
        nk_dict: material nk
        swarm_size, iterations: PSO params
        w, c1, c2: inertia, cognitive, social coefficients
        valid_thicknesses: list of valid thicknesses for quantization
        rng: numpy RandomState

    Returns:
        optimized structure dict
    """
    if rng is None:
        rng = np.random.RandomState()
    if valid_thicknesses is None:
        valid_thicknesses = list(range(10, 201, 10))

    materials = list(initial_structure['materials'])
    n_dims = len(materials)
    if n_dims == 0:
        return initial_structure

    base_thicknesses = np.array(initial_structure['thicknesses'], dtype=np.float64)

    # PSO bounds from vocabulary
    t_min = min(valid_thicknesses)
    t_max = max(valid_thicknesses)

    # Initialize particles
    positions = np.tile(base_thicknesses, (swarm_size, 1))
    positions += rng.uniform(-20, 20, (swarm_size, n_dims))
    positions = np.clip(positions, t_min, t_max)

    velocities = rng.uniform(-10, 10, (swarm_size, n_dims))

    p_best_pos = positions.copy()
    p_best_val = np.full(swarm_size, np.inf)

    g_best_pos = base_thicknesses.copy()
    g_best_val = np.inf

    def fitness_batch(positions_batch):
        """Evaluate fitness for a batch of positions."""
        results = []
        for pos in positions_batch:
            # Quantize for TMM
            quantized = []
            for t in pos:
                diffs = np.abs(np.array(valid_thicknesses) - t)
                quantized.append(valid_thicknesses[np.argmin(diffs)])
            f = _compute_fitness(materials, quantized, target_joint, nk_dict, theta=theta)
            results.append(_objective_value(f, objective))
        return np.array(results)

    for it in range(iterations):
        # Evaluate current positions
        vals = fitness_batch(positions)

        # Update personal bests
        improved = vals < p_best_val
        p_best_val[improved] = vals[improved]
        p_best_pos[improved] = positions[improved].copy()

        # Update global best
        min_idx = np.argmin(vals)
        if vals[min_idx] < g_best_val:
            g_best_val = vals[min_idx]
            g_best_pos = positions[min_idx].copy()

        # Update velocities and positions
        r1 = rng.uniform(0, 1, (swarm_size, n_dims))
        r2 = rng.uniform(0, 1, (swarm_size, n_dims))

        velocities = (w * velocities +
                       c1 * r1 * (p_best_pos - positions) +
                       c2 * r2 * (g_best_pos - positions))
        positions = positions + velocities
        positions = np.clip(positions, t_min, t_max)

    # Quantize final best
    final_thicknesses = []
    for t in g_best_pos:
        diffs = np.abs(np.array(valid_thicknesses) - t)
        final_thicknesses.append(valid_thicknesses[np.argmin(diffs)])

    final_fitness = _compute_fitness(materials, final_thicknesses, target_joint, nk_dict, theta=theta)

    return {
        'materials': materials,
        'thicknesses': final_thicknesses,
        'fitness': _objective_value(final_fitness, objective),
        'details': final_fitness,
        'pso_iterations': iterations,
        'pso_best_val': g_best_val,
    }


# ============================================================
# Main Perturbation Pipeline
# ============================================================

def perturb_structures(prepared_data, nk_dict, method='GA_PSO',
                        allowed_materials=None, valid_thicknesses=None,
                        seed=42, theta=THETA_DEG,
                        min_improvement=1e-4, max_joint_error=0.15,
                        min_high_t_worst_pol=0.8):
    """
    Run GA+PSO perturbation on prepared candidate structures.
    """
    valid_methods = {'GA', 'PSO', 'GA_PSO'}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    rng = np.random.RandomState(seed)

    if allowed_materials is None:
        allowed_materials = ALLOWED_MATERIALS
    if valid_thicknesses is None:
        valid_thicknesses = list(range(10, 201, 10))

    results = []

    for item in tqdm(prepared_data, desc="Perturbing"):
        target = item['target_spec_joint']
        original = {
            'materials': item['best_materials'],
            'thicknesses': item['best_thicknesses'],
        }
        objective = item.get('ranking_objective', 'joint_error')
        original_score = item.get('ranking_score')
        if original_score is None:
            original_score = _objective_value(item, objective)

        best_result = None
        best_error = float('inf')

        if method in ('GA', 'GA_PSO'):
            ga_result = ga_optimize(
                original, target, nk_dict,
                population_size=40, generations=20, mutation_rate=0.3,
                allowed_materials=allowed_materials,
                valid_thicknesses=valid_thicknesses,
                rng=rng, theta=theta, objective=objective,
            )
            if ga_result['fitness'] < best_error and ga_result['details'] is not None:
                best_error = ga_result['fitness']
                best_result = ga_result

        if method in ('PSO', 'GA_PSO'):
            pso_result = pso_optimize(
                original, target, nk_dict,
                swarm_size=25, iterations=30,
                valid_thicknesses=valid_thicknesses,
                rng=rng, theta=theta, objective=objective,
            )
            if pso_result['fitness'] < best_error:
                best_error = pso_result['fitness']
                best_result = pso_result

        if best_result is None or best_result.get('details') is None:
            continue

        # Check improvement exceeds minimum threshold
        improvement = original_score - best_error
        if improvement < max(MIN_IMPROVEMENT, min_improvement):
            continue

        # Build perturbed joint spectrum
        details = best_result['details']
        if details.get('E_joint', float('inf')) > max_joint_error:
            continue
        if (objective == 'high_transmission' and
                details.get('worst_pol_mean_T', -float('inf')) < min_high_t_worst_pol):
            continue
        pert_spec_joint = np.concatenate([
            np.asarray(details.get('sim_Rs', [])),
            np.asarray(details.get('sim_Ts', [])),
            np.asarray(details.get('sim_Rp', [])),
            np.asarray(details.get('sim_Tp', [])),
        ]).astype(np.float32)

        tokens = [f"{m}_{t}" for m, t in zip(
            best_result['materials'], best_result['thicknesses'])]

        results.append({
            'target_spec_joint': target,
            'original_struct_tokens': item['best_tokens'],
            'original_joint_error': item['E_joint'],
            'original_objective_score': original_score,
            'perturb_materials': best_result['materials'],
            'perturb_thicknesses': best_result['thicknesses'],
            'perturb_struct': tokens,
            'perturb_spec_joint': pert_spec_joint,
            'struct_hash': structure_hash_from_tokens(tokens),
            'new_joint_error': details.get('E_joint', float('nan')),
            'new_objective_score': best_error,
            'improvement': improvement,
            'mae_Rs': details.get('mae_Rs', float('nan')),
            'mae_Ts': details.get('mae_Ts', float('nan')),
            'mae_Rp': details.get('mae_Rp', float('nan')),
            'mae_Tp': details.get('mae_Tp', float('nan')),
            'E_s': details.get('E_s', float('nan')),
            'E_p': details.get('E_p', float('nan')),
            'E_joint': details.get('E_joint', float('nan')),
            'mean_Ts': details.get('mean_Ts', float('nan')),
            'mean_Tp': details.get('mean_Tp', float('nan')),
            'p05_Ts': details.get('p05_Ts', float('nan')),
            'p05_Tp': details.get('p05_Tp', float('nan')),
            'min_Ts': details.get('min_Ts', float('nan')),
            'min_Tp': details.get('min_Tp', float('nan')),
            'worst_pol_mean_T': details.get('worst_pol_mean_T', float('nan')),
            'high_T_loss': details.get('high_T_loss', float('nan')),
            'theta_deg': theta,
            'optimization_method': method,
            'optimization_objective': objective,
            'seed': seed,
        })

    return results


if __name__ == "__main__":
    print("perturb.py — Use via run.py (self_improving/run.py)")
