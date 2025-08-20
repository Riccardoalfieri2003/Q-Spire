# QELM Benchmarks

A centralized record of QELM training runs, organized by **simulation method**.
Each simulation has its own CSV that accumulates comparable runs (same columns, different values).

## Folder layout

```
Benchmark/
├─ README.md                 ← this file
├─ CPU/
│  └─ benchmarks.csv         ← CPU-only runs (e.g., July 100-vocab)
├─ Cubit/
│  └─ benchmarks.csv         ← Cubit-only runs (e.g., Aug 1000-vocab)
├─ GPU/                      ← (optional) add when you run GPU
│  └─ benchmarks.csv
├─ Hybrid/                   ← (optional)
│  └─ benchmarks.csv
├─ IBMQPU/                   ← (optional)
│  └─ benchmarks.csv
```

> **Why separate CSVs?**
> It keeps “apples with apples.” Different simulators (CPU/Cubit/GPU/Hybrid/QPU) can have very different performance and numeric behavior; per-simulator CSVs make trends obvious and comparisons fair.

## CSV schema (all simulators)

Each `benchmarks.csv` uses the same header so rows stay comparable across simulators:

```csv
run_id,date,sim_method,dataset,vocab_size,embedding_dim,heads,hidden_dim,learning_rate,epochs,threads,blocks,decimal_precision,entropy_factor,advanced_ansatz,data_reuploading,sub_bit_encoding,pauli_twirling,zne_enabled,zne_scaling,elapsed_seconds,loss,perplexity,notes,artifacts_model,artifacts_tokens
```

**Column meanings**

* **run\_id**: short name (e.g., `july_cpu_v100_e1`).
* **date**: ISO date `YYYY-MM-DD`.
* **sim\_method**: `CPU`, `Cubit`, `GPU`, `Hybrid`, or `IBMQPU`.
* **dataset**: file or corpus name (e.g., `Science.txt`).
* **vocab\_size / embedding\_dim / heads / hidden\_dim / learning\_rate / epochs / threads**: model/training knobs used.
* **blocks / decimal\_precision / entropy\_factor**: advanced quantum settings.
* **advanced\_ansatz / data\_reuploading / sub\_bit\_encoding / pauli\_twirling / zne\_enabled**: `true|false`.
* **zne\_scaling**: string of scaling factors if used (e.g., `"1,3,5"`), else empty.
* **elapsed\_seconds**: wall-clock seconds reported by training.
* **loss / perplexity**: final metrics after the last epoch.
* **notes**: freeform (e.g., “first CPU baseline”).
* **artifacts\_model / artifacts\_tokens**: paths to the saved `.qelm` and token-map JSON for the run (relative repo paths recommended).

## Current entries

### CPU → `Benchmark/CPU/benchmarks.csv`

```csv
run_id,date,sim_method,dataset,vocab_size,embedding_dim,heads,hidden_dim,learning_rate,epochs,threads,blocks,decimal_precision,entropy_factor,advanced_ansatz,data_reuploading,sub_bit_encoding,pauli_twirling,zne_enabled,zne_scaling,elapsed_seconds,loss,perplexity,notes,artifacts_model,artifacts_tokens
july_cpu_v100_e1,2025-07-xx,CPU,Science.txt,100,4,2,4,0.5,1,18,4,4,0.0,true,true,true,false,false,,71726.67,4.601739,63.494500,"CPU baseline; real dataset","/Models/July.qelm","/Models/July_token_map.json"
```

### Cubit → `Benchmark/Cubit/benchmarks.csv`

```csv
run_id,date,sim_method,dataset,vocab_size,embedding_dim,heads,hidden_dim,learning_rate,epochs,threads,blocks,decimal_precision,entropy_factor,advanced_ansatz,data_reuploading,sub_bit_encoding,pauli_twirling,zne_enabled,zne_scaling,elapsed_seconds,loss,perplexity,notes,artifacts_model,artifacts_tokens
aug_cubit_v1000_e1,2025-08-xx,Cubit,Science.txt,1000,4,2,4,0.5,1,24,4,4,0.0,true,true,true,false,false,,69499.99,6.907050,600.196349,"Cubit sim; 1000-vocab","/Models/August.qelm","/Models/August_token_map.json"
```

## Interpreting results

* **Loss / Perplexity**: lower is better.
* **Elapsed seconds**: total wall time; useful for throughput comparisons at a fixed dataset/epoch.
* **Config flags** (e.g., advanced ansatz, reuploading, sub-bit): record them—these often explain shifts in convergence or stability.
* **ZNE & Pauli Twirling**: leave `zne_enabled=true` and fill `zne_scaling` only when you actually ran ZNE; mark `pauli_twirling=true` if enabled.
