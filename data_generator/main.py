from running import run_generate_envs_for_a_pair, optimization_fba
from loading import load_model_simple
from modeling import generate_random_environment
from interactions import search_specific_interactions

import argparse

def main():
    parser = argparse.ArgumentParser(description="Running metabolic modeling for a specific pair and growing into randomly generated environments.")
    parser.add_argument('--microbe1', type=str, default="HGT_models_Agora/model2.mat", help='Path to metabolic network of the 1st Microbe')
    parser.add_argument('--microbe2', type=str, default="HGT_models_Agora/model3.mat", help='Path to metabolic network of the 2nd Microbe')
    parser.add_argument('--n_environments', type=int, default=1000, help='Number of generated environments')
    parser.add_argument('--min_compounds', type=int, default=50, help='Minimal amount of available nutrients')
    parser.add_argument('--max_compounds', type=int, default=424, help='Maximal amount of available nutrients')
    parser.add_argument('--method', type=str, default='FBA', help='Optimization method')
    parser.add_argument('--interaction', type=list, default=["Cooperative"], help='Type of interaction')
    parser.add_argument('--max_per_type', type=int, default=2000, help='Amount of environments')
    parser.add_argument('--max_attempts', type=int, default=100000, help='Maximal amount of environments')

    args = parser.parse_args()


    # run_generate_envs_for_a_pair(
    #     args.microbe1,
    #     args.microbe2,
    #     args.n_environments,
    #     args.min_compounds,
    #     args.max_compounds
    # )
    
    # optimization_fba(
    #     args.microbe1, 
    #     args.microbe2, 
    #     args.n_environments,
    #     args.min_compounds,
    #     args.max_compounds,
    #     args.method
    #     )
    
    # Search for cooperative
    search_specific_interactions(
        args.microbe1, 
        args.microbe2, 
        target_interactions=args.interaction,
        max_per_type=args.max_per_type,
        max_attempts=args.max_attempts, 
        optimization_method=args.method
    )
    
    # # Search for all interaction types
    # search_specific_interactions(
    #     "model1.mat", "model2.mat", 
    #     target_interactions=["Cooperative", "Competitive", "Obligate PlusX", "Obligate XPlus"],
    #     max_per_type=250,
    #     max_attempts=15000,
    #     optimization_method='MOMA'
    # )

    
if __name__ == "__main__":
    main()
