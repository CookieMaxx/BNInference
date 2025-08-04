#main Boolean Network Inference class

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from itertools import combinations
import time
import logging
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Tuple, Optional

from static_methods import (_infer_single_gene_parallel_multi, _create_default_rule_static,
                            _calculate_rule_balance_score_static, _extract_boolean_rule_from_tree_static)
from export import export_to_sbml_qual, save_results

#boolean Network Inference with parallelization support
class BooleanNetworkInference:
    def __init__(self,
                 discretization_method='median',
                 max_regulators=3,
                 min_consistency=0.65,
                 correlation_threshold=0.1,
                 balance_penalty=0.1,
                 default_rule_threshold=0.5):

        self.discretization_method = discretization_method
        self.max_regulators = max_regulators
        self.min_consistency = min_consistency
        self.correlation_threshold = correlation_threshold
        self.balance_penalty = balance_penalty
        self.default_rule_threshold = default_rule_threshold

        #result storage
        self.boolean_rules = {}
        self.network_edges = []
        self.discretized_data = None
        self.gene_names = None
        self.failed_genes = []

        #setup cleaner logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    #discretization with threshold selection
    def discretize_expression_data(self, expression_data: pd.DataFrame) -> pd.DataFrame:

        self.logger.info(f"Discretizing {expression_data.shape[1]} genes using {self.discretization_method}")

        discretized = expression_data.copy()
        discretization_info = {}

        for gene in expression_data.columns:
            gene_expr = expression_data[gene]

            if self.discretization_method == 'median':
                threshold = gene_expr.median()
            elif self.discretization_method == 'mean':
                threshold = gene_expr.mean()
            elif self.discretization_method == 'adaptive':
                q25, q75 = gene_expr.quantile([0.25, 0.75])
                threshold = (q25 + q75) / 2
            else:
                threshold = gene_expr.median()

            discretized[gene] = (gene_expr > threshold).astype(int)

            # Store discretization info for analysis
            on_fraction = discretized[gene].mean()
            discretization_info[gene] = {
                'threshold': threshold,
                'on_fraction': on_fraction,
                'is_constant': on_fraction in [0.0, 1.0]
            }

        self.discretized_data = discretized
        self.gene_names = list(expression_data.columns)
        self.discretization_info = discretization_info

        # Warn about constant genes
        constant_genes = [g for g, info in discretization_info.items() if info['is_constant']]
        if constant_genes:
            self.logger.warning(f"Found {len(constant_genes)} constant genes: {constant_genes[:5]}...")

        self.logger.info(
            f"Discretization complete. Average ON fraction: {np.mean([info['on_fraction'] for info in discretization_info.values()]):.3f}")
        return discretized
    #regulator selection with filtering
    def find_potential_regulators(self, target_gene: str) -> List[str]:

        correlations = self.discretized_data.corr()[target_gene].abs()

        # Remove self-correlation and filter by threshold
        potential_regulators = correlations[
            (correlations.index != target_gene) &
            (correlations >= self.correlation_threshold)
            ].sort_values(ascending=False)

        # Filter out constant genes as regulators
        potential_regulators = potential_regulators[
            ~potential_regulators.index.isin([g for g, info in self.discretization_info.items() if info['is_constant']])
        ]

        # Limit to reasonable number
        max_candidates = min(20, len(potential_regulators))
        return list(potential_regulators.head(max_candidates).index)

    #calculate balance score to penalize rules that favor all-off patterns
    def calculate_rule_balance_score(self, rule_str: str, predictions: np.array, actual: np.array) -> float:
        on_predictions = np.sum(predictions == 1)
        off_predictions = np.sum(predictions == 0)
        total_predictions = len(predictions)

        # calculate balance
        balance = min(on_predictions, off_predictions) / total_predictions

        #penelize rules with extreme imbalance
        balance_score = 1.0 - abs(0.5 - balance)

        #extract penalty for rules that are mostly NOT operations
        not_count = rule_str.count('NOT')
        total_vars = len(rule_str.split('AND')) + len(rule_str.split('OR'))
        if total_vars > 0:
            not_ratio = not_count / total_vars
            not_penalty = not_ratio * self.balance_penalty
        else:
            not_penalty = 0

        return balance_score - not_penalty

    #rule extraction with logic and balance checking
    def extract_boolean_rule_from_tree(self, tree, feature_names: List[str], target_gene: str) -> Optional[Dict]:

        tree_structure = tree.tree_

        if tree_structure.node_count == 1:
            #single node tree - check if it's meaningful
            prediction = int(tree_structure.value[0][0][1] > tree_structure.value[0][0][0])
            target_on_fraction = self.discretization_info[target_gene]['on_fraction']

            #accept constant rules only if they match actual distribution
            if (prediction == 1 and target_on_fraction > 0.7) or (prediction == 0 and target_on_fraction < 0.3):
                return {
                    'rule': str(prediction),
                    'variables': [],
                    'type': 'constant',
                    'is_trivial': True
                }
            else:
                return None

        #extract all possible rules from tree paths
        def extract_all_paths(node_id=0, path_conditions=[]):
            if tree_structure.children_left[node_id] == tree_structure.children_right[node_id]:
                #laefnode
                prediction = int(tree_structure.value[node_id][0][1] > tree_structure.value[node_id][0][0])
                return [(path_conditions.copy(), prediction)]

            paths = []
            feature_idx = tree_structure.feature[node_id]
            threshold = tree_structure.threshold[node_id]
            feature_name = feature_names[feature_idx]

            #left child (feature == 0)
            left_condition = f"NOT {feature_name}"
            paths.extend(extract_all_paths(
                tree_structure.children_left[node_id],
                path_conditions + [left_condition]
            ))

            #right child (feature == 1)
            right_condition = feature_name
            paths.extend(extract_all_paths(
                tree_structure.children_right[node_id],
                path_conditions + [right_condition]
            ))

            return paths

        all_paths = extract_all_paths()

        #find paths that predict activation (1)
        activation_paths = [path for path, prediction in all_paths if prediction == 1]

        if not activation_paths:
            return {
                'rule': '0',
                'variables': [],
                'type': 'always_off',
                'is_trivial': True
            }

        #convert paths to Boolean expressions
        if len(activation_paths) == 1:
            #single path - use AND
            conditions = activation_paths[0]
            if len(conditions) == 1:
                rule_str = conditions[0]
            else:
                rule_str = " AND ".join(conditions)
        else:
            #multiple paths - use OR of ANDs
            path_expressions = []
            for conditions in activation_paths[:3]:
                if len(conditions) == 1:
                    path_expressions.append(conditions[0])
                else:
                    path_expressions.append("(" + " AND ".join(conditions) + ")")
            rule_str = " OR ".join(path_expressions)

        #extract variables used
        variables = []
        for path in activation_paths:
            for condition in path:
                if condition.startswith("NOT "):
                    variables.append(condition[4:])
                else:
                    variables.append(condition)
        variables = list(set(variables))

        return {
            'rule': rule_str,
            'variables': variables,
            'type': 'decision_tree',
            'is_trivial': False
        }

    #create a default rule for genes that failed inference
    def create_default_rule(self, target_gene: str) -> Dict:
        target_on_fraction = self.discretization_info[target_gene]['on_fraction']

        if target_on_fraction > self.default_rule_threshold:
            rule = '1'
        else:
            rule = '0'

        return {
            'rule': rule,
            'variables': [],
            'consistency': target_on_fraction if rule == '1' else (1 - target_on_fraction),
            'f1_score': 0.0,  #low F1 indicates default rule
            'type': 'default',
            'is_trivial': True,
            'target_gene': target_gene
        }

    #inference with top-M rules per gene selection
    def infer_gene_regulation(self, target_gene: str, max_rules_per_gene: int = 3) -> Optional[List[Dict]]:
        self.logger.info(f"Processing {target_gene}")

        #check if target gene is constant
        if self.discretization_info[target_gene]['is_constant']:
            self.logger.warning(f"{target_gene} is constant - creating default rule")
            return [self.create_default_rule(target_gene)]

        #find potential regulators
        potential_regulators = self.find_potential_regulators(target_gene)

        if len(potential_regulators) == 0:
            self.logger.warning(f"No regulators found for {target_gene}")
            return [self.create_default_rule(target_gene)]

        #store ALL candidate rules for ranking
        all_candidate_rules = []

        # trying different combinations with scoring
        for num_regs in range(1, min(self.max_regulators + 1, len(potential_regulators) + 1)):
            for regulator_combo in combinations(potential_regulators, num_regs):
                X = self.discretized_data[list(regulator_combo)].values
                y = self.discretized_data[target_gene].values

                #train decision tree with better parameters
                tree = DecisionTreeClassifier(
                    max_depth=min(3, num_regs + 1),
                    min_samples_split=max(5, len(y) // 10),
                    min_samples_leaf=max(2, len(y) // 20),
                    class_weight='balanced',  # Handle imbalanced data
                    random_state=42
                )

                tree.fit(X, y)
                y_pred = tree.predict(X)

                #calculate metrics
                consistency = accuracy_score(y, y_pred)
                f1 = f1_score(y, y_pred, zero_division=0)

                #lower threshold for candidate collection (was self.min_consistency)
                if consistency >= 0.35:  # Much more permissive for candidate collection
                    #extract rule
                    rule_info = self.extract_boolean_rule_from_tree(tree, list(regulator_combo), target_gene)

                    if rule_info and not rule_info.get('is_trivial', False):
                        #calculate balance score
                        balance_score = self.calculate_rule_balance_score(rule_info['rule'], y_pred, y)

                        combined_score = f1 + balance_score + (consistency * 0.5)

                        rule_info.update({
                            'consistency': consistency,
                            'f1_score': f1,
                            'balance_score': balance_score,
                            'combined_score': combined_score,
                            'target_gene': target_gene,
                            'meets_original_threshold': consistency >= self.min_consistency,
                            'quality_tier': 'high' if (consistency >= 0.8 and f1 >= 0.8) else
                            'medium' if (consistency >= 0.6 and f1 >= 0.6) else 'low'
                        })

                        all_candidate_rules.append(rule_info)

        #seleciton of top M rules per gene
        if all_candidate_rules:
            #sorting by combined score (descending)
            all_candidate_rules.sort(key=lambda x: x['combined_score'], reverse=True)

            top_rules = all_candidate_rules[:max_rules_per_gene]

            #log selected rules
            for i, rule in enumerate(top_rules):
                self.logger.info(
                    f"{target_gene} Rule {i + 1}: {rule['rule']} "
                    f"(C:{rule['consistency']:.3f}, F1:{rule['f1_score']:.3f}, "
                    f"Tier:{rule['quality_tier']})")

            return top_rules
        else:
            self.logger.warning(f"No good rules found for {target_gene} - using default")
            self.failed_genes.append(target_gene)
            return [self.create_default_rule(target_gene)]

    #main inference method with handling and multiple rules per gene
    def infer_network(self, expression_data: pd.DataFrame, genes_subset: Optional[List[str]] = None,
                      max_rules_per_gene: int = 3) -> Dict:
        start_time = time.time()

        #dscitretise data
        self.discretize_expression_data(expression_data)

        #determine genes to analyze
        if genes_subset is None:
            target_genes = self.gene_names
        else:
            target_genes = [g for g in genes_subset if g in self.gene_names]

        self.logger.info(f"Starting inference for {len(target_genes)} genes (max {max_rules_per_gene} rules per gene)")

        #reset tracking
        self.failed_genes = []
        successful_inferences = 0
        total_rules = 0

        #infer regulation for each target gene
        for i, target_gene in enumerate(target_genes):
            if (i + 1) % 10 == 0:  # Progress update every 10 genes
                self.logger.info(f"Progress: {i + 1}/{len(target_genes)} genes processed")

            rules_list = self.infer_gene_regulation(target_gene, max_rules_per_gene)

            if rules_list is not None:
                #store multiple rules per gene
                self.boolean_rules[target_gene] = rules_list
                total_rules += len(rules_list)

                #add edges to network for all non-trivial rules
                non_trivial_rules = [r for r in rules_list if not r.get('is_trivial', False)]
                if non_trivial_rules:
                    successful_inferences += 1

                    for rule in non_trivial_rules:
                        for regulator in rule['variables']:
                            edge_type = 'activation'
                            if f'NOT {regulator}' in rule['rule']:
                                edge_type = 'inhibition'

                            self.network_edges.append({
                                'source': regulator,
                                'target': target_gene,
                                'type': edge_type,
                                'confidence': rule['f1_score'],
                                'rule_id': f"{target_gene}_rule_{rules_list.index(rule) + 1}",
                                'quality_tier': rule.get('quality_tier', 'unknown'),
                                'meets_original_threshold': rule.get('meets_original_threshold', False)
                            })

                #track failed genes (those with only default rules)
                if all(r.get('type') == 'default' for r in rules_list):
                    self.failed_genes.append(target_gene)

        end_time = time.time()

        #generate summary with discretization info
        results = {
            'boolean_rules': self.boolean_rules,
            'network_edges': self.network_edges,
            'failed_genes': self.failed_genes,
            'discretization_info': getattr(self, 'discretization_info', {}),
            'summary': {
                'total_genes_analyzed': len(target_genes),
                'successful_inferences': successful_inferences,
                'default_rules': len(self.failed_genes),
                'success_rate': successful_inferences / len(target_genes),
                'total_edges': len(self.network_edges),
                'total_rules': total_rules,
                'avg_rules_per_gene': total_rules / len(target_genes),
                'computation_time': end_time - start_time
            }
        }

        self.logger.info(
            f"Inference complete: {successful_inferences}/{len(target_genes)} successful ({results['summary']['success_rate']:.1%})")
        self.logger.info(f"Total rules: {total_rules}, Avg per gene: {results['summary']['avg_rules_per_gene']:.1f}")
        self.logger.info(f"Time: {end_time - start_time:.1f}s, Edges: {len(self.network_edges)}")

        return results

    #parallelized network inference method with multiple rules per gene
    def infer_network_parallel(self, expression_data: pd.DataFrame, genes_subset: Optional[List[str]] = None,
                               n_processes: int = None, max_rules_per_gene: int = 3) -> Dict:

        start_time = time.time()

        # Discretize data first (this needs to be done before parallelization)
        self.discretize_expression_data(expression_data)

        #determine genes to analyze
        if genes_subset is None:
            target_genes = self.gene_names
        else:
            target_genes = [g for g in genes_subset if g in self.gene_names]

        #determine number of processes
        if n_processes is None:
            n_processes = min(mp.cpu_count(), len(target_genes))

        self.logger.info(f"Starting parallel inference for {len(target_genes)} genes using {n_processes} processes")
        self.logger.info(f"Max rules per gene: {max_rules_per_gene}")

        #reset tracking
        self.failed_genes = []

        #create a partial function with the fixed parameters
        inference_func = partial(_infer_single_gene_parallel_multi,
                                 discretized_data=self.discretized_data,
                                 discretization_info=self.discretization_info,
                                 max_regulators=self.max_regulators,
                                 min_consistency=self.min_consistency,
                                 correlation_threshold=self.correlation_threshold,
                                 balance_penalty=self.balance_penalty,
                                 default_rule_threshold=self.default_rule_threshold,
                                 max_rules_per_gene=max_rules_per_gene)

        #usage of multiprocessing pool
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(inference_func, target_genes)

        #processing results
        self.boolean_rules = {}
        self.network_edges = []
        successful_inferences = 0
        total_rules = 0

        for rules_list in results:
            if rules_list is not None and len(rules_list) > 0:
                target_gene = rules_list[0]['target_gene']
                self.boolean_rules[target_gene] = rules_list
                total_rules += len(rules_list)

                # Add edges to network for all non-trivial rules
                non_trivial_rules = [r for r in rules_list if not r.get('is_trivial', False)]
                if non_trivial_rules:
                    successful_inferences += 1

                    for rule in non_trivial_rules:
                        for regulator in rule['variables']:
                            edge_type = 'activation'
                            if f'NOT {regulator}' in rule['rule']:
                                edge_type = 'inhibition'

                            self.network_edges.append({
                                'source': regulator,
                                'target': target_gene,
                                'type': edge_type,
                                'confidence': rule['f1_score'],
                                'rule_id': f"{target_gene}_rule_{rules_list.index(rule) + 1}",
                                'quality_tier': rule.get('quality_tier', 'unknown'),
                                'meets_original_threshold': rule.get('meets_original_threshold', False)
                            })

                #track failed genes (those with only default rules)
                if all(r.get('type') == 'default' for r in rules_list):
                    self.failed_genes.append(target_gene)

        end_time = time.time()

        #generate summary
        results = {
            'boolean_rules': self.boolean_rules,
            'network_edges': self.network_edges,
            'failed_genes': self.failed_genes,
            'discretization_info': self.discretization_info,
            'summary': {
                'total_genes_analyzed': len(target_genes),
                'successful_inferences': successful_inferences,
                'default_rules': len(self.failed_genes),
                'success_rate': successful_inferences / len(target_genes),
                'total_edges': len(self.network_edges),
                'total_rules': total_rules,
                'avg_rules_per_gene': total_rules / len(target_genes),
                'computation_time': end_time - start_time,
                'processes_used': n_processes
            }
        }

        self.logger.info(
            f"Parallel inference complete: {successful_inferences}/{len(target_genes)} successful ({results['summary']['success_rate']:.1%})")
        self.logger.info(f"Total rules: {total_rules}, Avg per gene: {results['summary']['avg_rules_per_gene']:.1f}")
        self.logger.info(
            f"Time: {end_time - start_time:.1f}s using {n_processes} processes, Edges: {len(self.network_edges)}")

        return results

    #static method assignments for parallel processing
    _infer_single_gene_parallel_multi = staticmethod(_infer_single_gene_parallel_multi)
    _create_default_rule_static = staticmethod(_create_default_rule_static)
    _calculate_rule_balance_score_static = staticmethod(_calculate_rule_balance_score_static)
    _extract_boolean_rule_from_tree_static = staticmethod(_extract_boolean_rule_from_tree_static)

    def export_to_sbml_qual(self, results: Dict, output_file: str, model_id: str = "ClostridiumBooleanNetwork"):
        return export_to_sbml_qual(self, results, output_file, model_id)

    def save_results(self, results: Dict, output_prefix: str):
        save_results(self, results, output_prefix)