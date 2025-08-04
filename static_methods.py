#static methods used for parallel processing

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from itertools import combinations
from typing import Dict, List, Optional

#static version of create_default_rule for multiprocessing
def _create_default_rule_static(target_gene: str, discretization_info: dict, default_rule_threshold: float) -> Dict:
    target_on_fraction = discretization_info[target_gene]['on_fraction']

    if target_on_fraction > default_rule_threshold:
        rule = '1'
    else:
        rule = '0'

    return {
        'rule': rule,
        'variables': [],
        'consistency': target_on_fraction if rule == '1' else (1 - target_on_fraction),
        'f1_score': 0.0,
        'type': 'default',
        'is_trivial': True,
        'target_gene': target_gene
    }

#static version for multiprocessing
def _calculate_rule_balance_score_static(rule_str: str, predictions: np.array, actual: np.array,
                                         balance_penalty: float) -> float:
    on_predictions = np.sum(predictions == 1)
    off_predictions = np.sum(predictions == 0)
    total_predictions = len(predictions)

    balance = min(on_predictions, off_predictions) / total_predictions
    balance_score = 1.0 - abs(0.5 - balance)

    not_count = rule_str.count('NOT')
    total_vars = len(rule_str.split('AND')) + len(rule_str.split('OR'))
    if total_vars > 0:
        not_ratio = not_count / total_vars
        not_penalty = not_ratio * balance_penalty
    else:
        not_penalty = 0

    return balance_score - not_penalty


def _extract_boolean_rule_from_tree_static(tree, feature_names: List[str], target_gene: str,
                                           discretization_info: dict) -> Optional[Dict]:
    """Static version of rule extraction for multiprocessing"""
    tree_structure = tree.tree_

    #hanlde single‐node tree
    if tree_structure.node_count == 1:
        prediction = int(tree_structure.value[0][0][1] > tree_structure.value[0][0][0])
        on_frac = discretization_info[target_gene]['on_fraction']
        if (prediction == 1 and on_frac > 0.7) or (prediction == 0 and on_frac < 0.3):
            return {'rule': str(prediction),
                    'variables': [],
                    'type': 'constant',
                    'is_trivial': True}
        return None

    #recursivly collect all (conds, prediction) paths
    def extract_all_paths(node_id=0, conds=None):
        conds = conds or []
        left = tree_structure.children_left[node_id]
        right = tree_structure.children_right[node_id]
        # leaf?
        if left == right:
            pred = int(tree_structure.value[node_id][0][1] > tree_structure.value[node_id][0][0])
            return [(conds.copy(), pred)]
        feat = feature_names[tree_structure.feature[node_id]]
        # 0‐branch = NOT feat
        paths = extract_all_paths(left, conds + [f"NOT {feat}"])
        # 1‐branch = feat
        paths += extract_all_paths(right, conds + [feat])
        return paths

    all_paths = extract_all_paths()
    #only those that predict ON
    activation = [p for p, pr in all_paths if pr == 1]
    if not activation:
        return {'rule': '0',
                'variables': [],
                'type': 'always_off',
                'is_trivial': True}

    #build boolean expression
    if len(activation) == 1:
        terms = activation[0]
        rule_str = terms[0] if len(terms) == 1 else " AND ".join(terms)
    else:
        or_terms = []
        for terms in activation[:3]:  # Limit complexity
            expr = terms[0] if len(terms) == 1 else "(" + " AND ".join(terms) + ")"
            or_terms.append(expr)
        rule_str = " OR ".join(or_terms)

    #gather unique vars (strip NOT)
    vars_ = {t[4:] if t.startswith("NOT ") else t
             for path in activation for t in path}

    return {
        'rule': rule_str,
        'variables': list(vars_),
        'type': 'decision_tree',
        'is_trivial': False
    }

#stat method for inferring multiple rules for a single gene (for multiprocessing)
def _infer_single_gene_parallel_multi(target_gene: str, discretized_data: pd.DataFrame,
                                      discretization_info: dict, max_regulators: int,
                                      min_consistency: float, correlation_threshold: float,
                                      balance_penalty: float, default_rule_threshold: float,
                                      max_rules_per_gene: int = 3) -> Optional[List[Dict]]:

    #if gene is constant, return default rule
    if discretization_info[target_gene]['is_constant']:
        return [_create_default_rule_static(
            target_gene, discretization_info, default_rule_threshold)]

    #identify potential regulators
    correlations = discretized_data.corr()[target_gene].abs()
    candidates = correlations[(correlations.index != target_gene) &
                              (correlations >= correlation_threshold)]
    #filtering out constant regulators
    constant_regs = {g for g, info in discretization_info.items() if info['is_constant']}
    potential_regulators = [g for g in candidates.sort_values(ascending=False).index
                            if g not in constant_regs][:20]

    #if no regulators, default
    if not potential_regulators:
        return [_create_default_rule_static(
            target_gene, discretization_info, default_rule_threshold)]

    all_candidate_rules = []
    y = discretized_data[target_gene].values

    #explore combinations of regulators
    for num_regs in range(1, min(max_regulators, len(potential_regulators)) + 1):
        for combo in combinations(potential_regulators, num_regs):
            X = discretized_data[list(combo)].values
            tree = DecisionTreeClassifier(
                max_depth=min(3, num_regs + 1),
                min_samples_split=max(5, len(y) // 10),
                min_samples_leaf=max(2, len(y) // 20),
                class_weight='balanced',
                random_state=42
            )
            tree.fit(X, y)
            y_pred = tree.predict(X)

            consistency = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, zero_division=0)

            # Candidate threshold
            if consistency >= 0.35:
                rule_info = _extract_boolean_rule_from_tree_static(
                    tree, list(combo), target_gene, discretization_info)
                if rule_info and not rule_info.get('is_trivial', False):
                    balance_score = _calculate_rule_balance_score_static(
                        rule_info['rule'], y_pred, y, balance_penalty)
                    combined = f1 + balance_score + (consistency * 0.5)
                    rule_info.update({
                        'consistency': consistency,
                        'f1_score': f1,
                        'balance_score': balance_score,
                        'combined_score': combined,
                        'target_gene': target_gene,
                        'meets_original_threshold': consistency >= min_consistency,
                        'quality_tier': ('high' if consistency >= 0.8 and f1 >= 0.8 else
                                         'medium' if consistency >= 0.6 and f1 >= 0.6 else
                                         'low')
                    })
                    all_candidate_rules.append(rule_info)

    #select top rules or default
    if all_candidate_rules:
        all_candidate_rules.sort(key=lambda r: r['combined_score'], reverse=True)
        return all_candidate_rules[:max_rules_per_gene]
    else:
        return [_create_default_rule_static(
            target_gene, discretization_info, default_rule_threshold)]