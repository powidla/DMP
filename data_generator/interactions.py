#!/usr/bin/env python3
"""
Targeted Interaction Search for Microbial Pairs
Efficiently searches for environments that produce specific interaction types
Only saves data for environments with target interactions
Files are organized into separate folders by interaction type
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import csc_matrix, csr_matrix, hstack
from scipy.optimize import linprog
import json
import os
from datetime import datetime
import random
from tqdm import tqdm
from collections import defaultdict

from loading import load_model_simple
from modeling import generate_random_environment, FriendOrFoeDataCollector, test_pair_growth_in_environment_flexible, create_pair_model_simple

def classify_interaction_detailed(m1_change, m2_change, tolerance=1e-16):
    '''
    Interaction classification using exact specified categories
    
    Parameters:
    - m1_change: growth change for microbe 1 
    - m2_change: growth change for microbe 2
    - tolerance: threshold for considering change significant (default 0.001)
    
    Returns:
    - interaction_type: interaction classification
    - interaction_category: broader category for grouping
    '''
    
    
    if m1_change > tolerance and m2_change > tolerance:
        interaction_type = "Cooperative"
        category = "Mutualism"
    elif m1_change < -tolerance and m2_change < -tolerance:
        interaction_type = "Competitive"
        category = "Competition"
    elif abs(m1_change) < tolerance and abs(m2_change) < tolerance:
        interaction_type = "Obligate XX"
        category = "Obligate"
    elif m1_change > tolerance and m2_change < -tolerance:
        interaction_type = "Obligate PlusX"
        category = "Obligate"
    elif m1_change < -tolerance and m2_change > tolerance:
        interaction_type = "Obligate XPlus"
        category = "Obligate"
    else:
        interaction_type = "Neutral"
        category = "Neutral"
    
    return interaction_type, category

class TargetedInteractionSearcher:
    """
    Searches for environments that produce specific microbial interactions
    Only collects and saves data for environments with target interactions
    Organizes output files into folders by interaction type
    """
    
    def __init__(self, microbe1, microbe2, target_interactions=None, 
                 max_environments_per_type=1000, interaction_tolerance=1e-16):
        """
        Initialize targeted searcher
        
        Parameters:
        - microbe1, microbe2: loaded microbe models
        - target_interactions: list of interaction types to search for
        - max_environments_per_type: maximum environments to save per interaction type
        - interaction_tolerance: threshold for interaction classification
        """
        
        self.microbe1 = microbe1
        self.microbe2 = microbe2
        self.target_interactions = target_interactions or [
            "Cooperative", "Competitive", "Obligate XX", 
            "Obligate PlusX", "Obligate XPlus", "Neutral"
        ]
        self.max_per_type = max_environments_per_type
        self.tolerance = interaction_tolerance

        self.found_interactions = defaultdict(list)
        self.interaction_counts = defaultdict(int)
        
        # Data 
        self.collector = FriendOrFoeDataCollector(microbe1, microbe2)
        
        print(f"Targeting interactions: {self.target_interactions}")
        print(f"Max environments per type: {self.max_per_type}")
    
    def is_target_complete(self):
        """Check if we've found enough environments for all target interactions"""
        for interaction_type in self.target_interactions:
            if len(self.found_interactions[interaction_type]) < self.max_per_type:
                return False
        return True
    
    def get_search_progress(self):
        '''
        Get current search progress
        '''
        progress = {}
        for interaction_type in self.target_interactions:
            found = len(self.found_interactions[interaction_type])
            target = self.max_per_type
            progress[interaction_type] = f"{found}/{target}"
        return progress
    
    def search_for_target_interactions(self, max_attempts=50000, min_nutrients=200, max_nutrients=424,
                                     optimization_method='FBA', min_growth_rate=1e-16):
        '''
        Search for environments that produce target interactions
        
        Parameters:
        - max_attempts: maximum environments to test
        - min_nutrients, max_nutrients: nutrient range per environment
        - optimization_method: FBA, MOMA, etc.
        - min_growth_rate: minimum viable growth rate
        
        Returns:
        - summary of found interactions
        '''
        
        print(f"\nTARGETED INTERACTION SEARCH")
        print("=" * 60)
        print(f"Max attempts: {max_attempts:,}")
        print(f"Optimization method: {optimization_method}")
        print(f"Nutrient range: {min_nutrients}-{max_nutrients}")
        print("=" * 60)
        
        # Setup
        num_compounds = self.microbe1["S_ext"].shape[0]
        pair_model = create_pair_model_simple(self.microbe1, self.microbe2)
        
        attempts = 0
        successful_tests = 0
        
        with tqdm(total=max_attempts, desc="Searching environments") as pbar:
            
            while attempts < max_attempts and not self.is_target_complete():
                attempts += 1
                env_id = f"search_{attempts:06d}"
                
                # Generate random environment
                env_rhslb, available_nutrients = generate_random_environment(
                    num_compounds, min_nutrients, max_nutrients
                )
                
                # Test pair growth
                success, results = test_pair_growth_in_environment_flexible(
                    self.microbe1, self.microbe2, env_rhslb, pair_model,
                    optimization_method=optimization_method,
                    min_growth_rate=min_growth_rate
                )
                
                if success:
                    successful_tests += 1
                    
                    # Classify interaction
                    m1_change = results['changes']['m1_change']
                    m2_change = results['changes']['m2_change']
                    
                    interaction_type, category = classify_interaction_detailed(
                        m1_change, m2_change, self.tolerance
                    )
                    
                    if (interaction_type in self.target_interactions and 
                        len(self.found_interactions[interaction_type]) < self.max_per_type):
                        
                        
                        environment_data = {
                            'env_id': env_id,
                            'env_rhslb': env_rhslb,
                            'available_nutrients': available_nutrients,
                            'results': results,
                            'interaction_type': interaction_type,
                            'interaction_category': category,
                            'm1_change': m1_change,
                            'm2_change': m2_change,
                            'n_nutrients': len(available_nutrients)
                        }
                        
                        self.found_interactions[interaction_type].append(environment_data)
                        
                        # Add to data collector
                        self.collector.add_environment_result(
                            env_id, env_rhslb, available_nutrients, results, pair_model
                        )
                    
                    self.interaction_counts[interaction_type] += 1
                
                # Update progress
                if attempts % 1000 == 0:
                    progress = self.get_search_progress()
                    success_rate = successful_tests / attempts * 100
                    
                    progress_str = " | ".join([f"{k}: {v}" for k, v in progress.items()])
                    pbar.set_postfix_str(f"Success: {success_rate:.1f}% | {progress_str}")
                
                pbar.update(1)
        
        print(f"\nSEARCH COMPLETED!")
        print(f"Attempts: {attempts:,}")
        print(f"Successful tests: {successful_tests:,} ({successful_tests/attempts*100:.1f}%)")
        
        print(f"\nFOUND TARGET INTERACTIONS:")
        total_found = 0
        for interaction_type in self.target_interactions:
            found = len(self.found_interactions[interaction_type])
            total_found += found
            print(f"   {interaction_type}: {found}/{self.max_per_type}")
        
        print(f"\nALL INTERACTION STATISTICS:")
        for interaction_type, count in sorted(self.interaction_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / successful_tests * 100 if successful_tests > 0 else 0
            print(f"   {interaction_type}: {count} ({percentage:.1f}%)")
        
        return {
            'attempts': attempts,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / attempts,
            'found_interactions': dict(self.found_interactions),
            'interaction_counts': dict(self.interaction_counts),
            'total_environments_saved': total_found
        }
    
    def create_interaction_folders(self, base_output_dir):
        """
        Create separate folders for each interaction type
        
        Parameters:
        - base_output_dir: base directory for all output
        
        Returns:
        - dictionary mapping interaction types to their folder paths
        """
        interaction_folders = {}
        
        # Create base directory
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Create folders for each interaction type that has data
        for interaction_type in self.found_interactions.keys():
            if len(self.found_interactions[interaction_type]) > 0:
                # Clean folder name (remove spaces, special characters)
                folder_name = interaction_type.lower().replace(' ', '_').replace('+', 'plus').replace('x', 'x')
                folder_path = os.path.join(base_output_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                interaction_folders[interaction_type] = folder_path
                
        return interaction_folders
    
    def save_targeted_results(self, output_dir="./targeted_interactions_output"):
        """
        Save targeted interaction results organized into folders by interaction type
        Creates separate folders for each interaction type
        """
        
        # Create base output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model1_clean = self.microbe1['name'].replace('.xml', '').replace(' ', '_')
        model2_clean = self.microbe2['name'].replace('.xml', '').replace(' ', '_')
        
        # Create interaction-specific folders
        interaction_folders = self.create_interaction_folders(output_dir)
        
        saved_files = {}
        interaction_summaries = {}
        
        print(f"\nSaving targeted interaction data to organized folders...")
        
        # Save consolidated data to base directory first
        if len(self.collector.summary_data) > 0:
            files = self.collector.save_consolidated_csvs(
                output_dir, timestamp, model1_clean, model2_clean
            )
            saved_files.update(files)
        
        # Process each interaction type separately
        for interaction_type, environments in self.found_interactions.items():
            if len(environments) > 0:
                
                # Get the folder for this interaction type
                interaction_folder = interaction_folders[interaction_type]
                
                print(f"Saving {interaction_type} data to: {os.path.basename(interaction_folder)}/")
                
                summary_data = []
                environment_details = []
                
                for env_data in environments:
                    summary_row = {
                        'env_id': env_data['env_id'],
                        'interaction_type': env_data['interaction_type'],
                        'interaction_category': env_data['interaction_category'],
                        'n_nutrients': env_data['n_nutrients'],
                        'm1_solo_growth': env_data['results']['growth_rates']['m1_alone'],
                        'm2_solo_growth': env_data['results']['growth_rates']['m2_alone'],
                        'm1_pair_growth': env_data['results']['growth_rates']['m1_with_m2_nw'],
                        'm2_pair_growth': env_data['results']['growth_rates']['m2_with_m1_nw'],
                        'm1_change': env_data['m1_change'],
                        'm2_change': env_data['m2_change'],
                        'm1_change_percent': (env_data['m1_change'] / env_data['results']['growth_rates']['m1_alone'] * 100) 
                                           if env_data['results']['growth_rates']['m1_alone'] > 0 else 0,
                        'm2_change_percent': (env_data['m2_change'] / env_data['results']['growth_rates']['m2_alone'] * 100) 
                                           if env_data['results']['growth_rates']['m2_alone'] > 0 else 0
                    }
                    summary_data.append(summary_row)
                    
                    # Environment details
                    env_detail = {
                        'env_id': env_data['env_id'],
                        'n_nutrients': env_data['n_nutrients'],
                        'available_nutrients': env_data['available_nutrients']
                    }
                    environment_details.append(env_detail)
                
                # Save interaction-specific summary to its folder
                summary_df = pd.DataFrame(summary_data)
                summary_filename = f"summary_{interaction_type}_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
                summary_path = os.path.join(interaction_folder, summary_filename)
                summary_df.to_csv(summary_path, index=False)
                saved_files[f'{interaction_type}_summary'] = os.path.join(os.path.basename(interaction_folder), summary_filename)
                
                # Save environment details to its folder
                env_details_df = pd.DataFrame(environment_details)
                env_filename = f"environments_{interaction_type}_{model1_clean}_vs_{model2_clean}_{timestamp}.csv"
                env_path = os.path.join(interaction_folder, env_filename)
                env_details_df.to_csv(env_path, index=False)
                saved_files[f'{interaction_type}_environments'] = os.path.join(os.path.basename(interaction_folder), env_filename)
                
                # Save interaction-specific metadata
                interaction_metadata = {
                    'interaction_type': interaction_type,
                    'count': len(environments),
                    'timestamp': timestamp,
                    'microbe1_name': self.microbe1['name'],
                    'microbe2_name': self.microbe2['name'],
                    'avg_m1_change': np.mean([env['m1_change'] for env in environments]),
                    'avg_m2_change': np.mean([env['m2_change'] for env in environments]),
                    'avg_nutrients': np.mean([env['n_nutrients'] for env in environments]),
                    'min_nutrients': min([env['n_nutrients'] for env in environments]),
                    'max_nutrients': max([env['n_nutrients'] for env in environments]),
                    'tolerance_used': self.tolerance
                }
                
                metadata_filename = f"metadata_{interaction_type}_{timestamp}.json"
                metadata_path = os.path.join(interaction_folder, metadata_filename)
                
                # Convert numpy types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return obj
                
                with open(metadata_path, 'w') as f:
                    json.dump(convert_numpy(interaction_metadata), f, indent=2)
                saved_files[f'{interaction_type}_metadata'] = os.path.join(os.path.basename(interaction_folder), metadata_filename)
                
                # Store summary for master file
                interaction_summaries[interaction_type] = {
                    'count': len(environments),
                    'avg_m1_change': np.mean([env['m1_change'] for env in environments]),
                    'avg_m2_change': np.mean([env['m2_change'] for env in environments]),
                    'avg_nutrients': np.mean([env['n_nutrients'] for env in environments]),
                    'min_nutrients': min([env['n_nutrients'] for env in environments]),
                    'max_nutrients': max([env['n_nutrients'] for env in environments]),
                    'folder': os.path.basename(interaction_folder)
                }
        
        # Save master summary in base directory
        master_summary = {
            'timestamp': timestamp,
            'microbe1_name': self.microbe1['name'],
            'microbe2_name': self.microbe2['name'],
            'target_interactions': self.target_interactions,
            'max_environments_per_type': self.max_per_type,
            'interaction_tolerance': self.tolerance,
            'found_interactions_summary': interaction_summaries,
            'total_environments_saved': sum(len(envs) for envs in self.found_interactions.values()),
            'saved_files': saved_files,
            'folder_structure': {interaction_type: os.path.basename(folder) 
                               for interaction_type, folder in interaction_folders.items()}
        }
        
        master_file = os.path.join(output_dir, f"MASTER_targeted_search_summary_{model1_clean}_vs_{model2_clean}_{timestamp}.json")
        with open(master_file, 'w') as f:
            json.dump(convert_numpy(master_summary), f, indent=2)
        
        print(f"\n Files saved in organized structure:")
        print(f" Base directory: {output_dir}")
        print(f" Master summary: {os.path.basename(master_file)}")
        
        for interaction_type, folder in interaction_folders.items():
            count = len(self.found_interactions[interaction_type])
            print(f" {os.path.basename(folder)}/ ({count} environments)")
            print(f"      ├── summary_{interaction_type}_*.csv")
            print(f"      ├── environments_{interaction_type}_*.csv")
            print(f"      └── metadata_{interaction_type}_*.json")
        
        return saved_files, master_summary

def search_specific_interactions(model1_path, model2_path, 
                                target_interactions=["Cooperative", "Competitive"],
                                max_per_type=500, max_attempts=20000,
                                optimization_method='FBA',
                                output_dir="./targeted_interactions"):
    '''
    Convenience function to search for specific interactions
    Files will be organized into separate folders by interaction type
    
    Parameters:
    - model1_path, model2_path: paths to model files
    - target_interactions: list of interaction types to find
    - max_per_type: max environments to save per interaction type
    - max_attempts: max environments to test
    - optimization_method: optimization method to use
    - output_dir: base directory for organized output folders
    
    Returns:
    - search results and saved file information
    '''
    
    print(f"TARGETED INTERACTION SEARCH")
    print(f"Targets: {target_interactions}")
    print(f"Goal: {max_per_type} environments per interaction type")
    print(f"Output: Files will be organized by interaction type in {output_dir}")
    
    # Load models
    microbe1 = load_model_simple(model1_path)
    microbe2 = load_model_simple(model2_path)
    
    # Initialize searcher
    searcher = TargetedInteractionSearcher(
        microbe1, microbe2, 
        target_interactions=target_interactions,
        max_environments_per_type=max_per_type
    )
    
    # Perform search
    search_results = searcher.search_for_target_interactions(
        max_attempts=max_attempts,
        optimization_method=optimization_method
    )
    
    # Save results in organized folders
    saved_files, summary = searcher.save_targeted_results(output_dir)
    
    return search_results, saved_files, summary

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# if __name__ == "__main__":
    
#     # Example 1: Search for cooperative interactions only
#     # Files will be saved to: ./targeted_interactions/cooperative/
#     results1 = search_specific_interactions(
#         "model1.mat", "model2.mat",
#         target_interactions=["Cooperative"],
#         max_per_type=1000,
#         max_attempts=10000,
#         output_dir="./cooperative_search_results"
#     )
    
#     # Example 2: Search for all interaction types
#     # Files will be organized into folders: ./multi_interaction_search/cooperative/, ./multi_interaction_search/competitive/, etc.
#     results2 = search_specific_interactions(
#         "model1.mat", "model2.mat", 
#         target_interactions=["Cooperative", "Competitive", "Obligate PlusX", "Obligate XPlus"],
#         max_per_type=250,
#         max_attempts=15000,
#         optimization_method='MOMA',
#         output_dir="./multi_interaction_search"
#     )
    
#     # Example 3: Focus on obligate interactions
#     # Files will be organized into: ./obligate_interactions/obligate_xx/, ./obligate_interactions/obligate_plusx/, etc.
#     results3 = search_specific_interactions(
#         "model1.mat", "model2.mat",
#         target_interactions=["Obligate XX", "Obligate PlusX", "Obligate XPlus"],
#         max_per_type=500,
#         max_attempts=25000,
#         output_dir="./obligate_interactions"
#     )