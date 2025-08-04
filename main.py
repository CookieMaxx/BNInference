#main script for Boolean Network Inference

from inference import BooleanNetworkInference
from data_utils import load_and_prepare_data, select_genes_for_analysis

if __name__ == "__main__":
    print("=" * 60)
    print("Boolean Network Inference for Clostridium beijerinckii")
    print("=" * 60)

    #loading data
    csv_file = "all_samples_fractional_counts.csv"
    expression_df = load_and_prepare_data(csv_file)

    #select 100 most variable genes for initial testing
    selected_genes_100 = select_genes_for_analysis(
        expression_df,
        n_genes=100, #can be adjusted to wanted gene selection
        selection_method='variance'  #use most variable genes
    )

    #selection of subset of genes for initial analysis
    print("\n" + "=" * 40)
    print(" Testing with 100 genes")
    print("=" * 40)

    #initialize the Boolean Network inference engine
    print(f"\nInitializing Boolean Network Inference...")
    bn = BooleanNetworkInference(
        discretization_method='mean',
        max_regulators=4,  # allow up to 4 regulators per gene
        min_consistency=0.60,  #slightly lower for more edges
        correlation_threshold=0.05,  # lower threshold for more potential regulators
        balance_penalty=0.15  #penalty for unbalanced rules
    )

    #run inference on 100 genes (parallel)
    print(f"\nRunning parallel inference on {len(selected_genes_100)} selected genes...")
    results_100 = bn.infer_network_parallel(
        expression_df,
        genes_subset=selected_genes_100,
        n_processes=6  #can be adjusted to how many CPU cores are available
    )

    #saving 100 gene results
    print(f"\nSaving results for 100-gene analysis...")
    bn.save_results(results_100, "clostridium_100genes")

    #printing summary for 100 genes
    print(f"\n" + "=" * 50)
    print("100-GENE ANALYSIS SUMMARY")
    print("=" * 50)
    summary = results_100['summary']
    print(f"Genes analyzed: {summary['total_genes_analyzed']}")
    print(f"Successful inferences: {summary['successful_inferences']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Total regulatory edges: {summary['total_edges']}")
    print(f"Computation time: {summary['computation_time']:.1f} seconds")
    print(f"Processes used: {summary['processes_used']}")
    print(f"Output files: clostridium_100genes_*")

    #giving option to run full dataset
    #used to test smaller subset first for parameter optimisation if needed
    print(f"\n" + "=" * 50)
    print("FULL DATASET ANALYSIS")
    print("=" * 50)

    run_full = input(f"Run analysis on all {expression_df.shape[1]} genes? (y/n): ").lower().strip()

    if run_full == 'y':
        print(f"Running parallel inference on ALL {expression_df.shape[1]} genes...")
        print("This may take significantly longer...")

        #create new instance for full analysis
        full_bn = BooleanNetworkInference(
            discretization_method='adaptive',
            max_regulators=4,
            min_consistency=0.60,
            correlation_threshold=0.05,
            balance_penalty=0.15
        )

        # Run full analysis
        results_full = full_bn.infer_network_parallel(
            expression_df,
            n_processes=6
        )

        #saving full bn results
        print(f"\nSaving results for full dataset analysis...")
        full_bn.save_results(results_full, "clostridium_full_dataset")

        #printing full summary
        print(f"\n" + "=" * 50)
        print("FULL DATASET ANALYSIS SUMMARY")
        print("=" * 50)
        summary_full = results_full['summary']
        print(f"Total genes analyzed: {summary_full['total_genes_analyzed']}")
        print(f"Successful inferences: {summary_full['successful_inferences']}")
        print(f"Success rate: {summary_full['success_rate']:.1%}")
        print(f"Total regulatory edges: {summary_full['total_edges']}")
        print(f"Computation time: {summary_full['computation_time']:.1f} seconds")
        print(f"Processes used: {summary_full['processes_used']}")
        print(f"Output files: clostridium_full_dataset_*")

    else:
        print("Skipping full dataset analysis.")
        print("You can run the full analysis later by changing 'run_full' to 'y'")

    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Check the generated files:")
    print("- *_boolean_rules.txt: Boolean rules for each gene")
    print("- *_network_edges.txt: Network edges (regulatory relationships)")
    print("- *_clostridium_bn.xml: SBML-qual format for systems biology tools")
    print("- *_analysis_summary.txt: Detailed analysis report")
    print("=" * 60)