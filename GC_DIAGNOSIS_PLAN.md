# GC Performance Diagnosis and Fix Plan

## Problem Summary
GC collection time is scaling exponentially (7s → 10s → 15s → 300s by gen 500) when calling `gc.collect()` every 20 simulation cycles. This indicates accumulating circular references or object retention issues.

## Root Cause Hypotheses

### 1. **NeuronGene.out_synapses List Accumulation** (HIGH PRIORITY)
**Issue**: When `SynapseGene.__init__()` is called, it calls `self.outof.connect(self)`, which appends the synapse to `NeuronGene.out_synapses`. During crossover, new `SynapseGene` objects are created and connected to existing `NeuronGene` objects, but old synapses may remain in the lists.

**Evidence**:
- `SynapseGene.__init__()` line 158: `self.outof.connect(self)`
- `NeuronGene.connect()` line 132-133: `self.out_synapses.append(synapse)`
- During crossover, new synapses are created but old ones may not be cleared

**Impact**: Each generation creates new synapses, but old synapses accumulate in `out_synapses` lists, creating:
- Growing lists that slow down iteration
- Circular references (NeuronGene ↔ SynapseGene)
- Objects that survive multiple GC cycles (Gen 2)

### 2. **Network.genome Reference Retention** (MEDIUM PRIORITY)
**Issue**: `Network.__init__()` stores `self.genome = genome` (line 525). Networks are created each generation but may retain references to old genomes.

**Evidence**:
- `Network.__init__()` line 525: `self.genome = genome`
- Networks are created fresh each generation in `simulate()` line 214
- Old Network objects may not be fully deallocated if references persist

### 3. **Species Progenitor Deepcopy Accumulation** (MEDIUM PRIORITY)
**Issue**: Species dictionary stores `deepcopy(genome.synapse_gene)` as progenitors. These deepcopies may retain references to old NeuronGene objects.

**Evidence**:
- `Simulation.__init__()` line 99: `deepcopy(genome.synapse_gene)`
- `mutate_and_speciate()` line 183: `deepcopy(genome.synapse_gene)`
- Deepcopies of synapse lists may include references to NeuronGene objects with accumulated `out_synapses`

### 4. **Circular Reference Chains** (HIGH PRIORITY)
**Issue**: Multiple circular reference patterns:
- `NeuronGene` → `out_synapses: list[SynapseGene]` → `SynapseGene` → `outof: NeuronGene`
- `Network` → `genome: NetworkGenome` → `synapse_gene: list[SynapseGene]` → `SynapseGene` → `outof: NeuronGene` → `out_synapses: list[SynapseGene]`

**Impact**: Python's reference counting can't free these objects, requiring cycle detection (expensive).

## Diagnosis Steps

### Step 1: Track Object Counts Over Time
**Goal**: Identify which object types are accumulating

**Implementation**:
```python
import sys
from collections import defaultdict

def count_objects_by_type():
    """Count objects by type in memory"""
    counts = defaultdict(int)
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        counts[obj_type] += 1
    return dict(counts)

# Call before and after each generation
# Log: NetworkGenome, Network, NeuronGene, SynapseGene, Neuron, Synapse counts
```

**Expected Output**: Track counts of NetworkGenome, Network, NeuronGene, SynapseGene, Neuron, Synapse objects across generations.

### Step 2: Inspect NeuronGene.out_synapses Growth
**Goal**: Verify if `out_synapses` lists are growing unbounded

**Implementation**:
```python
def analyze_neuron_synapse_counts(genomes):
    """Analyze synapse counts per neuron across genomes"""
    stats = {
        'total_neurons': 0,
        'total_synapse_refs': 0,
        'neurons_with_excess_synapses': 0,
        'max_synapses_per_neuron': 0
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
                print(f"Neuron {neuron.id} has {synapse_count} refs but only {actual_synapses} actual synapses")
    
    return stats
```

**Expected Output**: If `neurons_with_excess_synapses > 0`, this confirms accumulation.

### Step 3: Use `gc.get_referrers()` to Find Retention Chains
**Goal**: Identify what's keeping old objects alive

**Implementation**:
```python
def find_object_retainers(obj, max_depth=3, visited=None):
    """Find what objects are retaining a given object"""
    if visited is None:
        visited = set()
    
    if id(obj) in visited or max_depth == 0:
        return []
    
    visited.add(id(obj))
    referrers = gc.get_referrers(obj)
    results = []
    
    for ref in referrers:
        if ref is not obj:
            ref_type = type(ref).__name__
            results.append((ref_type, id(ref)))
            results.extend(find_object_retainers(ref, max_depth-1, visited))
    
    return results

# Sample old NetworkGenome objects and see what's retaining them
```

### Step 4: Profile GC Collection by Generation
**Goal**: Understand what GC is collecting

**Implementation**:
```python
# Enhanced GC callback
def gc_callback(phase, info):
    if phase == "stop":
        gen = info['generation']
        collected = info['collected']
        uncollectable = info.get('uncollectable', 0)
        
        # Log object types being collected
        if collected > 1000:  # Only log significant collections
            print(f"Gen {gen}: Collected {collected} objects, {uncollectable} uncollectable")
```

### Step 5: Check for Reference Cycles
**Goal**: Count cycles in object graph

**Implementation**:
```python
def count_cycles():
    """Count reference cycles in the object graph"""
    cycles = gc.get_objects()
    # Use tracemalloc or objgraph to visualize cycles
    import objgraph
    objgraph.show_most_common_types(limit=20)
    objgraph.show_backrefs([sample_genome], max_depth=3)
```

## Fix Strategy

### Fix 1: Clear NeuronGene.out_synapses During Crossover (CRITICAL)
**Problem**: New synapses are added but old ones aren't cleared when creating child genomes.

**Solution**: Clear `out_synapses` lists when creating new NeuronGene objects during crossover.

**Location**: `NetworkGenome.from_crossover()` method

**Implementation**:
```python
# In NeuronGene.crossover(), after creating new neuron:
new_neuron = NeuronGene(gene1.id, bias, gene1.activation_f)
new_neuron.out_synapses = []  # Clear the list - synapses will be reconnected
```

**Also needed**: When creating new SynapseGene objects, ensure they connect to fresh NeuronGene objects, not parent ones.

### Fix 2: Clear out_synapses When Creating New SynapseGene
**Problem**: When `SynapseGene.__init__()` calls `self.outof.connect(self)`, it adds to a list that may already contain old synapses.

**Solution**: Don't auto-connect in `__init__`. Instead, connect explicitly after all synapses are created, or clear the list first.

**Alternative**: Modify `SynapseGene.__init__()` to check if synapse already exists in list before adding.

### Fix 3: Weak Reference for Network.genome (OPTIONAL)
**Problem**: Network objects hold strong references to genomes.

**Solution**: Use `weakref` for the genome reference if it's only needed for fitness updates.

**Note**: May not be necessary if Networks are short-lived.

### Fix 4: Periodic Cleanup of NeuronGene.out_synapses
**Problem**: Even with fixes, lists may accumulate over many generations.

**Solution**: Add cleanup method to rebuild `out_synapses` from `synapse_gene`:

```python
def rebuild_synapse_connections(self):
    """Rebuild out_synapses lists from synapse_gene"""
    for neuron in self.all_neuron_genes:
        neuron.out_synapses = []
    
    for synapse in self.synapse_gene:
        synapse.outof.out_synapses.append(synapse)
```

Call this after crossover and mutation operations.

### Fix 5: Explicit Deletion of Old Objects
**Problem**: Python may not immediately free objects even without references.

**Solution**: Explicitly clear lists and set references to None:

```python
# In reproduce(), after creating next_gen:
for genome in self.genomes:
    # Clear references
    genome.input_neurons = None
    genome.output_neurons = None
    # ... etc
self.genomes = next_gen
```

## Implementation Priority

1. **IMMEDIATE**: Fix 1 - Clear out_synapses during crossover ✅ IMPLEMENTED
2. **IMMEDIATE**: Fix 2 - Fix SynapseGene connection logic ✅ IMPLEMENTED  
3. **HIGH**: Fix 4 - Add cleanup method and call it after operations ✅ IMPLEMENTED
4. **MEDIUM**: Implement diagnosis Step 2 to verify fixes
5. **LOW**: Fix 3 - Weak references (if needed)
6. **LOW**: Fix 5 - Explicit deletion (if needed)

## Implemented Fixes

### Fix: `rebuild_synapse_connections()` Method
**Location**: `NetworkGenome.rebuild_synapse_connections()`

**What it does**:
- Clears all `out_synapses` lists in all NeuronGene objects
- Rebuilds the lists from the current `synapse_gene` list
- Ensures `out_synapses` only contains synapses that are actually in `synapse_gene`

**Called at**:
1. `NetworkGenome.__init__()` - Only if `synapse_gene` is already populated (not during `from_crossover`)
2. `NetworkGenome.from_crossover()` - After all synapses are created for the child
3. `NetworkGenome.mutate()` - After all mutation operations complete

**Why this fixes the issue**:
- Prevents accumulation of old synapse references in `out_synapses` lists
- Breaks circular reference chains by ensuring lists only contain current synapses
- Ensures consistency between `synapse_gene` and `out_synapses` lists

## Testing Plan

1. Run simulation for 100 generations with fixes
2. Monitor GC collection time - should remain constant or grow linearly
3. Use diagnosis tools to verify object counts stabilize
4. Check that `out_synapses` counts match actual synapse counts
5. Profile memory usage over time

## Expected Outcomes

- GC collection time should remain constant (or grow linearly with network size)
- Object counts should stabilize after initial growth
- No accumulation of old synapses in `out_synapses` lists
- Memory usage should plateau rather than grow exponentially

