"""
Diagnostic script to identify GC performance issues.
Run this alongside your simulation to track object accumulation.
"""

import gc
import sys
from collections import defaultdict
from typing import Dict, Any

def count_objects_by_type() -> Dict[str, int]:
    """Count objects by type in memory"""
    counts = defaultdict(int)
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        counts[obj_type] += 1
    return dict(counts)

def analyze_neuron_synapse_counts(genomes):
    """Analyze synapse counts per neuron across genomes"""
    stats = {
        'total_neurons': 0,
        'total_synapse_refs': 0,
        'neurons_with_excess_synapses': 0,
        'max_synapses_per_neuron': 0,
        'excess_details': []
    }
    
    for genome in genomes:
        for neuron in genome.all_neuron_genes:
            stats['total_neurons'] += 1
            synapse_count = len(neuron.out_synapses)
            stats['total_synapse_refs'] += synapse_count
            if synapse_count > stats['max_synapses_per_neuron']:
                stats['max_synapses_per_neuron'] = synapse_count
            
            # Check if synapse count exceeds actual synapses in genome
            actual_synapses = sum(1 for s in genome.synapse_gene if s.outof.id == neuron.id)
            if synapse_count > actual_synapses:
                stats['neurons_with_excess_synapses'] += 1
                stats['excess_details'].append({
                    'neuron_id': neuron.id,
                    'ref_count': synapse_count,
                    'actual_synapses': actual_synapses,
                    'excess': synapse_count - actual_synapses
                })
    
    return stats

def get_gc_stats():
    """Get current GC statistics"""
    return {
        'gen0': len(gc.get_objects(generation=0)),
        'gen1': len(gc.get_objects(generation=1)),
        'gen2': len(gc.get_objects(generation=2)),
        'collections': gc.get_stats(),
    }

def find_sample_retainers(obj, max_depth=2):
    """Find what objects are retaining a given object (sample)"""
    try:
        referrers = gc.get_referrers(obj)
        ref_types = defaultdict(int)
        for ref in referrers[:10]:  # Sample first 10
            ref_type = type(ref).__name__
            ref_types[ref_type] += 1
        return dict(ref_types)
    except:
        return {}

def diagnose_generation(genomes, generation_num):
    """Run full diagnosis for a generation"""
    print(f"\n=== Generation {generation_num} Diagnosis ===")
    
    # Object counts
    obj_counts = count_objects_by_type()
    relevant_types = ['NetworkGenome', 'Network', 'NeuronGene', 'SynapseGene', 'Neuron', 'Synapse', 'Layer']
    print("\nObject counts:")
    for obj_type in relevant_types:
        if obj_type in obj_counts:
            print(f"  {obj_type}: {obj_counts[obj_type]}")
    
    # Neuron synapse analysis
    neuron_stats = analyze_neuron_synapse_counts(genomes)
    print(f"\nNeuron synapse stats:")
    print(f"  Total neurons: {neuron_stats['total_neurons']}")
    print(f"  Total synapse refs: {neuron_stats['total_synapse_refs']}")
    print(f"  Max synapses per neuron: {neuron_stats['max_synapses_per_neuron']}")
    print(f"  Neurons with excess synapses: {neuron_stats['neurons_with_excess_synapses']}")
    
    if neuron_stats['neurons_with_excess_synapses'] > 0:
        print(f"\n  ⚠️  WARNING: Found {neuron_stats['neurons_with_excess_synapses']} neurons with excess synapses!")
        print("  First 5 examples:")
        for detail in neuron_stats['excess_details'][:5]:
            print(f"    Neuron {detail['neuron_id']}: {detail['ref_count']} refs, {detail['actual_synapses']} actual, {detail['excess']} excess")
    
    # GC stats
    gc_stats = get_gc_stats()
    print(f"\nGC stats:")
    print(f"  Gen 0 objects: {gc_stats['gen0']}")
    print(f"  Gen 1 objects: {gc_stats['gen1']}")
    print(f"  Gen 2 objects: {gc_stats['gen2']}")
    
    # Sample retainers (if excess found)
    if neuron_stats['neurons_with_excess_synapses'] > 0:
        sample_genome = genomes[0]
        retainers = find_sample_retainers(sample_genome)
        if retainers:
            print(f"\nSample retainers for first genome:")
            for ref_type, count in retainers.items():
                print(f"  {ref_type}: {count}")
    
    return {
        'object_counts': obj_counts,
        'neuron_stats': neuron_stats,
        'gc_stats': gc_stats
    }

if __name__ == "__main__":
    print("GC Diagnostic Tool")
    print("Import this module and call diagnose_generation(genomes, gen_num) after each generation")

