#export and save functionality

import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from typing import Dict
from config import CORE_NS, QUAL_NS, MATH_NS

#MathML conversion with proper apply elements for operators
#xtract the MathML namespace (e.g. "{http://www.w3.org/1998/Math/MathML}")
def _convert_rule_to_mathml(rule: str, math_elem: ET.Element):
    ns_brace = math_elem.tag[:math_elem.tag.index('}') + 1]

    if rule == '0':
        ET.SubElement(math_elem, f"{ns_brace}false")
    elif rule == '1':
        ET.SubElement(math_elem, f"{ns_brace}true")
    elif ' OR ' in rule:
        #create apply element for OR operation
        apply_elem = ET.SubElement(math_elem, f"{ns_brace}apply")
        ET.SubElement(apply_elem, f"{ns_brace}or")

        for part in rule.split(' OR '):
            part = part.strip().strip('()')
            if ' AND ' in part:
                #create nested apply element for AND operation
                and_apply = ET.SubElement(apply_elem, f"{ns_brace}apply")
                ET.SubElement(and_apply, f"{ns_brace}and")

                for atom in part.split(' AND '):
                    atom = atom.strip()
                    if atom.startswith('NOT '):
                        not_apply = ET.SubElement(and_apply, f"{ns_brace}apply")
                        ET.SubElement(not_apply, f"{ns_brace}not")
                        ci = ET.SubElement(not_apply, f"{ns_brace}ci")
                        ci.text = atom[4:]
                    else:
                        ci = ET.SubElement(and_apply, f"{ns_brace}ci")
                        ci.text = atom
            else:
                #single term in OR
                if part.startswith('NOT '):
                    not_apply = ET.SubElement(apply_elem, f"{ns_brace}apply")
                    ET.SubElement(not_apply, f"{ns_brace}not")
                    ci = ET.SubElement(not_apply, f"{ns_brace}ci")
                    ci.text = part[4:]
                else:
                    ci = ET.SubElement(apply_elem, f"{ns_brace}ci")
                    ci.text = part

    elif ' AND ' in rule:
        #create apply element for AND operation
        apply_elem = ET.SubElement(math_elem, f"{ns_brace}apply")
        ET.SubElement(apply_elem, f"{ns_brace}and")

        for atom in rule.split(' AND '):
            atom = atom.strip()
            if atom.startswith('NOT '):
                not_apply = ET.SubElement(apply_elem, f"{ns_brace}apply")
                ET.SubElement(not_apply, f"{ns_brace}not")
                ci = ET.SubElement(not_apply, f"{ns_brace}ci")
                ci.text = atom[4:]
            else:
                ci = ET.SubElement(apply_elem, f"{ns_brace}ci")
                ci.text = atom

    elif rule.startswith('NOT '):
        #create apply element for NOT operation
        apply_elem = ET.SubElement(math_elem, f"{ns_brace}apply")
        ET.SubElement(apply_elem, f"{ns_brace}not")
        ci = ET.SubElement(apply_elem, f"{ns_brace}ci")
        ci.text = rule[4:]
    else:
        #variable reference
        ci = ET.SubElement(math_elem, f"{ns_brace}ci")
        ci.text = rule


def _create_transition(transitions_parent, gene, rule_info, transition_id, logger):
    #helper method to create a single transition element
    tr = ET.SubElement(transitions_parent, f"{{{QUAL_NS}}}transition", {
        'id': f"tr_{gene}",
        'name': f"Regulation of {gene}"
    })

    #inputs
    if rule_info['variables']:
        li = ET.SubElement(tr, f"{{{QUAL_NS}}}listOfInputs")
        for reg in rule_info['variables']:
            inp = ET.SubElement(li, f"{{{QUAL_NS}}}input", {
                'id': f"input_{reg}_to_{gene}",
                'qualitativeSpecies': reg,
                'transitionEffect': 'none',
                'sign': 'negative' if f'NOT {reg}' in rule_info['rule'] else 'positive'
            })
            logger.info(f"  {reg} -> {gene}: {inp.get('sign').upper()}")

    #outputs
    lo = ET.SubElement(tr, f"{{{QUAL_NS}}}listOfOutputs")
    ET.SubElement(lo, f"{{{QUAL_NS}}}output", {
        'id': f"output_{gene}",
        'qualitativeSpecies': gene,
        'transitionEffect': 'assignmentLevel'
    })

    #function terms
    lft = ET.SubElement(tr, f"{{{QUAL_NS}}}listOfFunctionTerms")
    #defaultTerm
    ET.SubElement(lft, f"{{{QUAL_NS}}}defaultTerm", {
        'resultLevel': '0'
    })
    #activation term
    if rule_info['rule'] not in ('0', '1'):
        ft = ET.SubElement(lft, f"{{{QUAL_NS}}}functionTerm", {
            'resultLevel': '1'
        })
        math = ET.SubElement(ft, f"{{{MATH_NS}}}math")
        #convert your textual rule into MathML under <math>
        _convert_rule_to_mathml(rule_info['rule'], math)

#saving XML
def _save_pretty_xml(element: ET.Element, filename: str):

    rough_string = ET.tostring(element, encoding='unicode')
    reparsed = minidom.parseString(rough_string)

    pretty_xml = reparsed.toprettyxml(indent="  ")

    #remove the first line and any empty lines
    lines = pretty_xml.split('\n')
    #skips first line if starts with <?xml
    if lines[0].strip().startswith('<?xml'):
        lines = lines[1:]

    #removing empty lines
    clean_lines = [line for line in lines if line.strip()]
    clean_xml = '\n'.join(clean_lines)

    #writing our own proper xml declaration
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(clean_xml)

#exporting boolean network into a qualified SBML file
def export_to_sbml_qual(inference_obj, results: Dict, output_file: str, model_id: str = "ClostridiumBooleanNetwork"):
    logger = inference_obj.logger
    logger.info(f"Exporting to SBML-qual: {output_file}")
    ET.register_namespace('', CORE_NS)
    ET.register_namespace('qual', QUAL_NS)
    ET.register_namespace('math', MATH_NS)

    sbml = ET.Element(f"{{{CORE_NS}}}sbml", {
        'level': '3',
        'version': '1',
        'qual:required': 'true'
    })

    model = ET.SubElement(sbml, f"{{{CORE_NS}}}model", {
        'id': model_id,
        'name': 'Boolean Network for Clostridium beijerinckii'
    })

    lc = ET.SubElement(model, f"{{{CORE_NS}}}listOfCompartments")
    ET.SubElement(lc, f"{{{CORE_NS}}}compartment", {
        'id': 'cell',
        'constant': 'true'
    })

    all_genes = set()
    for r in results['boolean_rules'].values():
        if isinstance(r, list):  # Multiple rules per gene
            for rule in r:
                if not rule.get('is_trivial', False):
                    all_genes.update(rule['variables'])
                    all_genes.add(rule['target_gene'])
        else:  # Single rule (backward compatibility)
            if not r.get('is_trivial', False):
                all_genes.update(r['variables'])
                all_genes.add(r['target_gene'])

    logger.info(f"Total species in network: {len(all_genes)}")

    lqs = ET.SubElement(model, f"{{{QUAL_NS}}}listOfQualitativeSpecies")
    for gene in sorted(all_genes):
        ET.SubElement(lqs, f"{{{QUAL_NS}}}qualitativeSpecies", {
            'id': gene,
            'name': gene,
            'compartment': 'cell',
            'constant': 'false',
            'maxLevel': '1'
        })

    lt = ET.SubElement(model, f"{{{QUAL_NS}}}listOfTransitions")
    transition_count = 0

    for gene, rules_info in results['boolean_rules'].items():
        if isinstance(rules_info, list):  # Multiple rules per gene
            non_trivial_rules = [r for r in rules_info if not r.get('is_trivial', False)]
            if non_trivial_rules:
                #for multiple rules, create one transition with best rule
                best_rule = max(non_trivial_rules, key=lambda x: x.get('combined_score', 0))
                _create_transition(lt, gene, best_rule, transition_count, logger)
                transition_count += 1
        else:
            if not rules_info.get('is_trivial', False):
                _create_transition(lt, gene, rules_info, transition_count, logger)
                transition_count += 1

    # validation log
    expected = len([r for r in results['boolean_rules'].values()
                    if (isinstance(r, list) and any(not rule.get('is_trivial', False) for rule in r)) or
                    (isinstance(r, dict) and not r.get('is_trivial', False))])
    if transition_count != expected:
        logger.warning(f"Transition count mismatch! Expected {expected}, created {transition_count}")
    else:
        logger.info(f"SBML validation: {transition_count} transitions created as expected")

    # write out
    _save_pretty_xml(sbml, output_file)
    logger.info(
        f"SBML-qual export complete: File={output_file}, Species={len(all_genes)}, Transitions={transition_count}")
    return transition_count, len(all_genes)

#saving results with formatting and filenames for multiple rules per gene
def save_results(inference_obj, results: Dict, output_prefix: str):
    logger = inference_obj.logger

    #boolean rules file - for multiple rules per gene
    rules_file = f"{output_prefix}_boolean_rules.txt"
    with open(rules_file, 'w') as f:
        f.write("# Boolean Network Rules for Clostridium beijerinckii\n")
        f.write("# Format: Target_Gene = Boolean_Expression\n")
        f.write(f"# Success rate: {results['summary']['success_rate']:.1%}\n")
        f.write(f"# Total edges: {results['summary']['total_edges']}\n")
        f.write(f"# Total rules: {results['summary']['total_rules']}\n")
        f.write(f"# Avg rules per gene: {results['summary']['avg_rules_per_gene']:.1f}\n\n")

        #process rules (multiple per gene)
        for target_gene, rules_list in results['boolean_rules'].items():
            if isinstance(rules_list, list):
                non_trivial_rules = [r for r in rules_list if not r.get('is_trivial', False)]
                if non_trivial_rules:
                    f.write(f"# === {target_gene} ===\n")
                    for i, rule_info in enumerate(non_trivial_rules):
                        f.write(f"{target_gene}_rule_{i + 1} = {rule_info['rule']}\n")
                        f.write(f"# Consistency: {rule_info['consistency']:.3f}, ")
                        f.write(f"F1: {rule_info['f1_score']:.3f}, ")
                        f.write(f"Balance: {rule_info.get('balance_score', 0):.3f}, ")
                        f.write(f"Tier: {rule_info.get('quality_tier', 'unknown')}\n")
                    f.write("\n")
            else:
                if not rules_list.get('is_trivial', False):
                    f.write(f"{target_gene} = {rules_list['rule']}\n")
                    f.write(f"# Consistency: {rules_list['consistency']:.3f}, ")
                    f.write(f"F1: {rules_list['f1_score']:.3f}, ")
                    f.write(f"Balance: {rules_list.get('balance_score', 0):.3f}\n\n")

        #failed genes section
        if results['failed_genes']:
            f.write(f"\n# Failed inferences ({len(results['failed_genes'])} genes):\n")
            f.write(f"# {', '.join(results['failed_genes'])}\n")

    #network edges file - with additional metadata
    edges_file = f"{output_prefix}_network_edges.txt"
    with open(edges_file, 'w') as f:
        f.write("Source\tTarget\tType\tConfidence\tRule_ID\tQuality_Tier\tMeets_Original_Threshold\n")
        for edge in results['network_edges']:
            f.write(f"{edge['source']}\t{edge['target']}\t{edge['type']}\t{edge['confidence']:.3f}\t"
                    f"{edge.get('rule_id', 'unknown')}\t{edge.get('quality_tier', 'unknown')}\t"
                    f"{edge.get('meets_original_threshold', False)}\n")

    #SBML-qual file with validation
    sbml_file = f"{output_prefix}_clostridium_bn.xml"
    transition_count, species_count = export_to_sbml_qual(inference_obj, results, sbml_file)

    #analysis summary - for multiple rules
    summary_file = f"{output_prefix}_analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Boolean Network Inference Analysis Summary\n")
        f.write("=" * 50 + "\n\n")

        summary = results['summary']
        f.write(f"Total genes analyzed: {summary['total_genes_analyzed']}\n")
        f.write(f"Successful inferences: {summary['successful_inferences']}\n")
        f.write(f"Default rules created: {summary['default_rules']}\n")
        f.write(f"Success rate: {summary['success_rate']:.1%}\n")
        f.write(f"Total regulatory edges: {summary['total_edges']}\n")
        f.write(f"Total rules generated: {summary['total_rules']}\n")
        f.write(f"Average rules per gene: {summary['avg_rules_per_gene']:.1f}\n")
        f.write(f"Computation time: {summary['computation_time']:.1f} seconds\n")
        if 'processes_used' in summary:
            f.write(f"Processes used: {summary['processes_used']}\n")
        f.write("\n")

        #rule quality analysis by tier
        f.write("RULE QUALITY ANALYSIS BY TIER\n")
        f.write("-" * 30 + "\n")

        tier_counts = {'high': 0, 'medium': 0, 'low': 0}
        original_threshold_count = 0
        total_non_trivial_rules = 0

        for gene, rules_list in results['boolean_rules'].items():
            if isinstance(rules_list, list):
                for rule in rules_list:
                    if not rule.get('is_trivial', False):
                        total_non_trivial_rules += 1
                        tier = rule.get('quality_tier', 'unknown')
                        if tier in tier_counts:
                            tier_counts[tier] += 1
                        if rule.get('meets_original_threshold', False):
                            original_threshold_count += 1
            else:
                if not rules_list.get('is_trivial', False):
                    total_non_trivial_rules += 1
                    tier = rules_list.get('quality_tier', 'unknown')
                    if tier in tier_counts:
                        tier_counts[tier] += 1
                    if rules_list.get('meets_original_threshold', False):
                        original_threshold_count += 1

        f.write(f"High quality rules (C>=0.8, F1>=0.8): {tier_counts['high']}\n")
        f.write(f"Medium quality rules (C>=0.6, F1>=0.6): {tier_counts['medium']}\n")
        f.write(f"Low quality rules (C<0.6 or F1<0.6): {tier_counts['low']}\n")
        f.write(f"Rules meeting original threshold: {original_threshold_count}\n")
        f.write(f"Total non-trivial rules: {total_non_trivial_rules}\n")

        #edge distribution analysis
        f.write(f"\nEDGE DISTRIBUTION ANALYSIS\n")
        f.write("-" * 25 + "\n")

        edge_types = {}
        quality_tiers = {}
        for edge in results['network_edges']:
            edge_type = edge.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

            tier = edge.get('quality_tier', 'unknown')
            quality_tiers[tier] = quality_tiers.get(tier, 0) + 1

        f.write(f"Activation edges: {edge_types.get('activation', 0)}\n")
        f.write(f"Inhibition edges: {edge_types.get('inhibition', 0)}\n")
        f.write(f"High quality edges: {quality_tiers.get('high', 0)}\n")
        f.write(f"Medium quality edges: {quality_tiers.get('medium', 0)}\n")
        f.write(f"Low quality edges: {quality_tiers.get('low', 0)}\n")

        if 'discretization_info' in results:
            disc_info = results['discretization_info']
            on_fractions = [info['on_fraction'] for info in disc_info.values()]

            f.write("DISCRETIZATION ANALYSIS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Average ON fraction: {np.mean(on_fractions):.3f}\n")
            f.write(f"Genes always OFF: {sum(1 for frac in on_fractions if frac == 0.0)}\n")
            f.write(f"Genes always ON: {sum(1 for frac in on_fractions if frac == 1.0)}\n")

            analyzed_genes = set(results['boolean_rules'].keys())
            analyzed_fractions = [disc_info[gene]['on_fraction'] for gene in analyzed_genes if gene in disc_info]
            balanced_count = sum(1 for frac in analyzed_fractions if 0.3 <= frac <= 0.7)
            f.write(f"Balanced genes (0.3-0.7): {balanced_count}\n")

            f.write(f"\nRECIPROCAL RELATIONSHIPS\n")
            f.write("-" * 24 + "\n")

            #finding reciprocal pairs
            reciprocal_pairs = []
            for edge1 in results['network_edges']:
                for edge2 in results['network_edges']:
                    if (edge1['source'] == edge2['target'] and
                            edge1['target'] == edge2['source'] and
                            edge1['source'] < edge1['target']):  # Avoid duplicates
                        reciprocal_pairs.append((edge1, edge2))

            if reciprocal_pairs:
                f.write(f"Found {len(reciprocal_pairs)} reciprocal regulatory pairs:\n")
                for edge1, edge2 in reciprocal_pairs:
                    f.write(f"  {edge1['source']} --{edge1['type']}--> {edge1['target']}\n")
                    f.write(f"  {edge2['source']} --{edge2['type']}--> {edge2['target']}\n")
                    f.write(f"  Confidences: {edge1['confidence']:.3f}, {edge2['confidence']:.3f}\n\n")
            else:
                f.write("No reciprocal regulatory pairs found\n")

            #showing example ON fractions for debugging
            f.write(f"\nExample gene ON fractions:\n")
            for i, (gene, info) in enumerate(list(disc_info.items())[:10]):
                if gene in analyzed_genes:
                    f.write(f"  {gene}: {info['on_fraction']:.3f}\n")
        else:
            f.write("DISCRETIZATION ANALYSIS\n")
            f.write("-" * 25 + "\n")
            f.write("No discretization info available\n")

    logger.info(f"Results saved: {rules_file}, {edges_file}, {sbml_file}, {summary_file}")