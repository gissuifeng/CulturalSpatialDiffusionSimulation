"""
model.py
Core simulation engine for the Cultural Spatiotemporal Diffusion Simulation.
(Multinomial Version)
"""
import os
import itertools
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Relative import within the core package
from .city import City

class CSDSModel:
    """
    Cultural Spatiotemporal Diffusion Simulation (CSDS) Engine.
    Handles data initialization, spatiotemporal migration, and cultural evolution processes.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the model with parsed YAML configuration.
        """
        self.config = config
        
        # Resolve absolute paths based on the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(project_root, config['paths']['data_in_dir'])
        self.output_dir = os.path.join(project_root, config['paths']['data_out_dir'])
        
        # Simulation parameters
        self.iterations = config['simulation']['iterations']
        
        # Hyperparameters
        hp = config['hyperparameters']
        self.transform_rate_H = hp['transform_rate_H']
        self.contact_rate = hp['contact_rate']
        self.transform_rate_I = hp['transform_rate_I']
        self.transform_rate_A = hp['transform_rate_A']
        self.itoA_rate = hp['itoA_rate']
        self.htoI_rate = hp['htoI_rate']
        
        # State containers
        self.cities: Dict[str, City] = {}
        self.flows: Dict[str, int] = {}             # Format: "O_D" -> flow_pop
        self.cultural_strength: Dict[str, float] = {} # Format: "C1_C2" -> strength
        self.valid_genes: List[str] = []
        
        # Output Tracking
        self.history_city_state: List[Dict] = []
        self.history_flow_state: List[Dict] = []

    def load_data(self):
        """Loads empirical input datasets and initializes the city agents."""
        logging.info("Loading empirical data into CSDS environment...")
        
        city_pop_df = pd.read_csv(os.path.join(self.data_dir, "city_pop.csv"), header=0)
        flow_df = pd.read_csv(os.path.join(self.data_dir, "city_pop_flow.csv"), header=0)
        strength_df = pd.read_csv(os.path.join(self.data_dir, "cultural_strength.csv"), header=0)
        culture_pop_df = pd.read_csv(os.path.join(self.data_dir, "city_culture_pop.csv"), header=0)

        for _, row in city_pop_df.iterrows():
            city_id, pop, culture, removed = str(row.iloc[0]), int(row.iloc[1]), row.iloc[2], int(row.iloc[3])
            self.cities[city_id] = City(city_id, pop, culture, removed)
            
        for _, row in flow_df.iterrows():
            self.flows[f"{row.iloc[0]}_{row.iloc[1]}"] = int(row.iloc[2])
            
        for _, row in strength_df.iterrows():
            self.cultural_strength[f"{row.iloc[0]}_{row.iloc[1]}"] = float(row.iloc[2])
            
        unique_cultures = set(city.culture_type for city in self.cities.values())
        for c in unique_cultures:
            self.valid_genes.extend([f"N_{c}_0", f"H_{c}_0"])
        self.valid_genes.append("R_0_0")
        for c1, c2 in itertools.combinations(list(unique_cultures), 2):
            self.valid_genes.extend([f"I_{c1}_{c2}", f"A_{c1}_{c2}"])
            
        for _, row in culture_pop_df.iterrows():
            city_id = str(row.iloc[0])
            if city_id in self.cities:
                for gene in self.valid_genes:
                    self.cities[city_id].cultural_structure[gene] = (
                        int(row[gene]) if gene in culture_pop_df.columns else 0
                    )

    def _get_gene_key(self, prefix: str, c1: str, c2: str) -> str:
        """Internal helper to resolve order-agnostic gene combinations."""
        k1 = f"{prefix}_{c1}_{c2}"
        if k1 in self.valid_genes: return k1
        return f"{prefix}_{c2}_{c1}"

    def simulate_migration(self) -> Dict[str, Dict[str, int]]:
        """Simulates inter-city migration utilizing high-performance Multinomial sampling."""
        flow_output = {od: {g: 0 for g in self.valid_genes} for od in self.flows.keys()}
        outflow_totals = {}
        destinations_map = {}
        
        for od, flow_pop in self.flows.items():
            o_city, d_city = od.split('_')
            outflow_totals[o_city] = outflow_totals.get(o_city, 0) + flow_pop
            destinations_map.setdefault(o_city, []).append((od, flow_pop))

        for city_id, city in self.cities.items():
            if city_id not in outflow_totals: continue
            
            total_out = outflow_totals[city_id]
            genes, probs = city.get_gene_distribution()
            
            # Vectorized multinomial sampling
            out_genes_counts = np.random.multinomial(total_out, probs)
            
            destinations = destinations_map[city_id]
            dest_probs = np.array([d[1] for d in destinations], dtype=float)
            dest_probs /= dest_probs.sum()
            
            for i, gene in enumerate(genes):
                if out_genes_counts[i] == 0: continue
                
                gene_dest_distribution = np.random.multinomial(out_genes_counts[i], dest_probs)
                for j, (od, _) in enumerate(destinations):
                    migrant_count = gene_dest_distribution[j]
                    if migrant_count > 0:
                        d_city = od.split('_')[1]
                        self.cities[city_id].cultural_structure[gene] -= migrant_count
                        self.cities[d_city].cultural_structure[gene] += migrant_count
                        flow_output[od][gene] += migrant_count
                        
        return flow_output

    def update_cultural_status(self):
        """Updates the immediate state of immigrants entering new cultural matrices."""
        for city_id, city in self.cities.items():
            native_gene = f"N_{city.culture_type}_0"
            native_h_gene = f"H_{city.culture_type}_0"
            
            for gene in list(city.cultural_structure.keys()):
                pop = city.cultural_structure[gene]
                if pop <= 0: continue
                
                if gene != native_gene and gene.startswith('N'):
                    _, origin_c, _ = gene.split('_')
                    new_h_gene = f"H_{origin_c}_0"
                    city.cultural_structure[new_h_gene] += pop
                    city.cultural_structure[gene] = 0
                    
                if gene == native_h_gene:
                    city.cultural_structure[native_gene] += pop
                    city.cultural_structure[gene] = 0

    def _get_cultural_antibody(self, source, target):
        ca = float(self.cultural_strength.get(f"{source}_{target}", 1.0))
        if np.isnan(ca) or ca <= 0:
            raise ValueError(f"CA must be positive for {source} -> {target}: {ca}")
        return ca

    def simulate_acculturation(self):
        """Executes the intra-city cultural contact and conversion dynamics."""
        for city_id, city in self.cities.items():
            native_culture = city.culture_type
            native_gene = f"N_{native_culture}_0"
            native_pop = city.cultural_structure.get(native_gene, 0)
            total_pop = sum(city.cultural_structure.values())
            
            if total_pop == 0: continue

            delta_N_loss = 0
            deltas: Dict[str, float] = {}
            
            def add_delta(gene_key, amount):
                deltas[gene_key] = deltas.get(gene_key, 0) + amount

            for gene, pop in city.cultural_structure.items():
                if pop <= 0: continue
                
                parts = gene.split('_')
                gene_type = parts[0]
                
                if gene_type == 'H':
                    hetero_culture = parts[1]
                    ca = self._get_cultural_antibody(hetero_culture, native_culture)
                    
                    p1 = self.transform_rate_H * self.contact_rate * native_pop * (pop / total_pop) / ca
                    i_gene = self._get_gene_key('I', native_culture, hetero_culture)
                    add_delta(i_gene, p1)
                    delta_N_loss += p1
                    
                    convert_H_to_I = pop * self.htoI_rate / ca
                    add_delta(i_gene, convert_H_to_I)
                    add_delta(gene, -convert_H_to_I)
                    
                elif gene_type == 'I':
                    c1, c2 = parts[1], parts[2]
                    p2 = self.transform_rate_I * self.contact_rate * native_pop * (pop / total_pop)
                    
                    target_c = c2 if native_culture == c1 else c1 if native_culture == c2 else np.random.choice([c1, c2])
                    ca = self._get_cultural_antibody(target_c, native_culture)
                    
                    i_gene = self._get_gene_key('I', native_culture, target_c)
                    add_delta(i_gene, p2 / ca)
                    delta_N_loss += p2 / ca
                    
                    convert_I_to_A = pop * self.itoA_rate / ca
                    a_gene = self._get_gene_key('A', c1, c2)
                    add_delta(a_gene, convert_I_to_A)
                    add_delta(gene, -convert_I_to_A)
                    
                elif gene_type == 'A':
                    c1, c2 = parts[1], parts[2]
                    p3 = self.transform_rate_A * self.contact_rate * native_pop * (pop / total_pop)
                    target_c = c2 if native_culture == c1 else c1 if native_culture == c2 else np.random.choice([c1, c2])
                    ca = self._get_cultural_antibody(target_c, native_culture)
                    
                    i_gene = self._get_gene_key('I', native_culture, target_c)
                    add_delta(i_gene, p3 / ca)
                    delta_N_loss += p3 / ca

            city.cultural_structure[native_gene] -= int(delta_N_loss)
            for g, change in deltas.items():
                city.cultural_structure[g] += int(change)

            city.pop = sum(city.cultural_structure.values())
            city.removed_pop = city.cultural_structure.get("R_0_0", 0)

    def run(self):
        """Main execution loop for the simulation."""
        for iteration in range(self.iterations):
            logging.info(f"--- Iteration {iteration + 1}/{self.iterations} ---")
            
            flow_output = self.simulate_migration()
            self.update_cultural_status()
            self.simulate_acculturation()
            
            for od, genes in flow_output.items():
                record = {"Iteration": iteration + 1, "O_D": od}
                record.update(genes)
                self.history_flow_state.append(record)
                
            for city_id, city in self.cities.items():
                record = {"Iteration": iteration + 1, "CityID": city_id}
                record.update(city.cultural_structure)
                self.history_city_state.append(record)

    def export_results(self):
        """Exports tracked simulation results to CSV format."""
        logging.info("Exporting results...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        pd.DataFrame(self.history_city_state).to_csv(
            os.path.join(self.output_dir, "country_output.csv"), index=False
        )
        pd.DataFrame(self.history_flow_state).to_csv(
            os.path.join(self.output_dir, "country_output_popflow.csv"), index=False
        )
        logging.info("Simulation successfully completed and exported.")