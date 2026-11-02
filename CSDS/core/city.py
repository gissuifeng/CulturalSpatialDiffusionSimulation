"""
city.py
Defines the core spatial entity (City Agent) in the CSDS framework.
"""
from typing import Dict, Tuple, List
import numpy as np

class City:
    """
    Represents a city agent in the spatial network, storing demographic and 
    cultural genetic structures.
    """
    def __init__(self, city_id: str, pop: int, culture_type: str, removed_pop: int):
        self.city_id = city_id
        self.pop = pop
        self.culture_type = culture_type
        self.removed_pop = removed_pop
        
        # Cultural structure dictionary: {gene_code: population_count}
        self.cultural_structure: Dict[str, int] = {}

    def get_gene_distribution(self) -> Tuple[List[str], np.ndarray]:
        """
        Calculates the probability distribution of cultural genes within the city.
        
        Returns:
            Tuple containing:
            - List of gene names (strings).
            - Numpy array of corresponding probabilities (float).
        """
        genes = list(self.cultural_structure.keys())
        populations = np.array(list(self.cultural_structure.values()), dtype=float)
        populations[populations < 0] = 0
        total_pop = populations.sum()
        
        if total_pop == 0:
            return genes, np.zeros_like(populations)
        
        probs = populations / total_pop
        probs = probs / probs.sum()
            
        return genes, probs