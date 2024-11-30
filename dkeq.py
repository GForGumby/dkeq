import numpy as np
import pandas as pd
from typing import List, Dict
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def load_players_from_csv(filename: str) -> dict:
    """
    Expected CSV format: dfs_id,pos,Salary,SS Proj,fd_std
    Returns numpy arrays for faster processing
    """
    df = pd.read_csv(filename)
    base_positions = ['PG', 'SG', 'SF', 'PF', 'C']
    pos_matrix = np.zeros((len(df), 8), dtype=np.int8)  # PG,SG,SF,PF,C,G,F,UTIL
    
    for i, pos_str in enumerate(df['pos']):
        positions = [p.strip() for p in str(pos_str).replace('/', ',').split(',')]
        for pos in positions:
            if pos in base_positions:
                pos_idx = base_positions.index(pos)
                pos_matrix[i, pos_idx] = 1
                
        # Set G eligibility (PG/SG)
        if pos_matrix[i, 0] or pos_matrix[i, 1]:
            pos_matrix[i, 5] = 1
            
        # Set F eligibility (SF/PF)
        if pos_matrix[i, 2] or pos_matrix[i, 3]:
            pos_matrix[i, 6] = 1
            
        # Set UTIL eligibility (all positions)
        pos_matrix[i, 7] = 1
                
    return {
        'ids': df['dfs_id'].values,
        'positions': pos_matrix,
        'salaries': df['Salary'].values.astype(np.int32),
        'means': df['SS Proj'].values.astype(np.float32),
        'stds': df['fd_std'].values.astype(np.float32)
    }

class FastDraftKingsNBA:
    SALARY_CAP = 50000
    ROSTER_SIZE = 8
    POS_REQUIREMENTS = np.array([1,1,1,1,1,1,1,1], dtype=np.int8)
    
    def __init__(self, player_data: dict, min_lineup_projection: float = 240.0):
        self.player_data = player_data
        self.num_players = len(player_data['ids'])
        self.min_lineup_projection = min_lineup_projection
        self.valid_players_by_pos = []
        
        # Pre-sort players by projection for each position
        for pos in range(8):
            valid_indices = np.where(player_data['positions'][:, pos] == 1)[0]
            sorted_indices = valid_indices[np.argsort(-player_data['means'][valid_indices])]
            self.valid_players_by_pos.append(sorted_indices)
        
    def generate_lineup_indices(self) -> np.ndarray:
        max_attempts = 1000
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            lineup = []
            used_players = set()
            current_proj = 0
            
            for pos in range(8):
                valid_players = [p for p in self.valid_players_by_pos[pos] if p not in used_players]
                if not valid_players:
                    break
                    
                player = random.choice(valid_players)
                lineup.append(player)
                used_players.add(player)
                current_proj += self.player_data['means'][player]
                
                # Early projection check
                remaining_slots = self.ROSTER_SIZE - len(lineup)
                if remaining_slots > 0:
                    min_needed_per_remaining = (self.min_lineup_projection - current_proj) / remaining_slots
                    if min_needed_per_remaining > 50:  # Adjust this threshold as needed
                        break
            
            if len(lineup) == self.ROSTER_SIZE and self._is_valid_lineup(np.array(lineup)):
                return np.array(lineup)
        
        # If we couldn't generate a valid lineup, try with lower minimum projection
        temp_min = self.min_lineup_projection
        self.min_lineup_projection *= 0.9  # Reduce by 10%
        lineup = self.generate_lineup_indices()  # Recursive call with lower minimum
        self.min_lineup_projection = temp_min  # Restore original minimum
        return lineup
    
    def _is_valid_lineup(self, indices: np.ndarray) -> bool:
        """Check salary cap and minimum projection"""
        if np.sum(self.player_data['salaries'][indices]) > self.SALARY_CAP:
            return False
            
        lineup_projection = np.sum(self.player_data['means'][indices])
        if lineup_projection < self.min_lineup_projection:
            return False
            
        return True

def generate_lineup_batch(dk, size: int) -> List[np.ndarray]:
    """Generate a batch of lineups"""
    return [dk.generate_lineup_indices() for _ in range(size)]

def generate_initial_field(dk, field_size: int, batch_size: int = 1000) -> np.ndarray:
    """Generate initial field using parallel processing"""
    num_cores = multiprocessing.cpu_count()
    batches = [batch_size] * (field_size // batch_size)
    if field_size % batch_size:
        batches.append(field_size % batch_size)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        lineups = list(executor.map(
            generate_lineup_batch,
            [dk] * len(batches),
            batches
        ))
    
    return np.vstack(lineups)

def simulate_field(lineup_indices: np.ndarray, player_data: dict, num_sims: int = 100):
    n_lineups = len(lineup_indices)
    scores = np.zeros((n_lineups, num_sims))
    
    means = player_data['means'][lineup_indices].sum(axis=1)
    stds = np.sqrt(np.sum(np.square(player_data['stds'][lineup_indices]), axis=1))
    
    for i in range(n_lineups):
        scores[i] = np.random.normal(means[i], stds[i], num_sims)
    
    return scores

def run_equilibrium_simulation(
    csv_filename: str,
    field_size: int,
    payout_structure: Dict[int, float],
    min_ev: float = 1.0,
    max_iterations: int = 10,
    sims_per_iter: int = 100,
    min_lineup_projection: float = 240.0
) -> tuple:
    player_data = load_players_from_csv(csv_filename)
    dk = FastDraftKingsNBA(player_data, min_lineup_projection=min_lineup_projection)
    
    print("Generating initial field...")
    current_field = generate_initial_field(dk, field_size)
    best_ev_lineups = None
    best_ev_values = None
    
    for iteration in range(max_iterations):
        print(f"Starting iteration {iteration + 1}")
        
        scores = simulate_field(current_field, player_data, sims_per_iter)
        
        ev = np.zeros(len(current_field))
        rankings = np.argsort(-scores, axis=0)
        
        for sim in range(sims_per_iter):
            for rank, lineup_idx in enumerate(rankings[:, sim], 1):
                if rank in payout_structure:
                    ev[lineup_idx] += payout_structure[rank]
        
        ev /= sims_per_iter
        
        plus_ev_mask = ev >= min_ev
        if best_ev_lineups is None or np.sum(plus_ev_mask) > 0:
            best_ev_lineups = current_field[plus_ev_mask]
            best_ev_values = ev[plus_ev_mask]
        
        plus_ev_lineups = current_field[plus_ev_mask]
        
        if len(plus_ev_lineups) == 0:
            print("No +EV lineups found")
            break
            
        print(f"Found {len(plus_ev_lineups)} +EV lineups")
        print(f"Max EV: {np.max(ev):.2f}, Min EV: {np.min(ev[plus_ev_mask]):.2f}")
        
        if len(plus_ev_lineups) == len(current_field):
            print("Field has converged")
            break
            
        new_field = np.zeros((field_size, dk.ROSTER_SIZE), dtype=np.int32)
        new_field[:len(plus_ev_lineups)] = plus_ev_lineups
        
        remaining = field_size - len(plus_ev_lineups)
        if remaining > 0:
            new_lineups = generate_initial_field(dk, remaining)
            new_field[len(plus_ev_lineups):] = new_lineups
            
        current_field = new_field
    
    return best_ev_lineups, player_data, best_ev_values

def export_lineups(filename: str, lineups: np.ndarray, player_data: dict, ev_values: np.ndarray):
    """Export +EV lineups sorted by EV"""
    sort_idx = np.argsort(-ev_values)
    sorted_lineups = lineups[sort_idx]
    lineup_ids = player_data['ids'][sorted_lineups]
    
    df = pd.DataFrame(lineup_ids, columns=[f'player_{i+1}' for i in range(8)])
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    payout_structure = {
        1: 1500.00,
        2: 750.00,
        3: 300.00,
        4: 150.00,
        5: 100.00,
        6: 75.00,
        7: 60.00, 8: 60.00,
        9: 50.00, 10: 50.00,
        11: 40.00, 12: 40.00, 13: 40.00, 14: 40.00,
        15: 30.00, 16: 30.00, 17: 30.00, 18: 30.00, 19: 30.00,
        20: 25.00, 21: 25.00, 22: 25.00, 23: 25.00, 24: 25.00, 25: 25.00,
        **dict.fromkeys(range(26, 36), 20.00),    # 26th-35th
        **dict.fromkeys(range(36, 46), 15.00),    # 36th-45th
        **dict.fromkeys(range(46, 61), 10.00),    # 46th-60th
        **dict.fromkeys(range(61, 81), 8.00),     # 61st-80th
        **dict.fromkeys(range(81, 106), 6.00),    # 81st-105th
        **dict.fromkeys(range(106, 161), 5.00),   # 106th-160th
        **dict.fromkeys(range(161, 276), 4.00),   # 161st-275th
        **dict.fromkeys(range(276, 551), 3.00),   # 276th-550th
        **dict.fromkeys(range(551, 1266), 2.00),  # 551st-1265th
        **dict.fromkeys(range(1266, 2716), 1.50), # 1266th-2715th
        **dict.fromkeys(range(2716, 8186), 1.00), # 2716th-8185th
    }
    
    final_lineups, player_data, ev_values = run_equilibrium_simulation(
        csv_filename="test proj.csv",
        field_size=40000,
        payout_structure=payout_structure,
        min_ev=8.0,
        max_iterations=50,
        sims_per_iter=100,
        min_lineup_projection=215.0
    )
    
    export_lineups(
        "optimal_lineups.csv", 
        final_lineups, 
        player_data,
        ev_values
    )
