# Task 2 review

Review range: `f4a2c65..46d84ad` in the copied inner repository. Reviewed the complete supplied diff, Task 2 brief and implementation report. Source, HEAD and index were not modified; this report is the only review artifact written.

## Strengths

- `src/kohakumoss/adapters/train/lori.py:127-177`: one CPU float32 concatenation supplies the global absolute-B quantile. Selection uses `>=`, retains ties and all-zero scores, and permits an individually empty layer. Flattened indices follow output-major B storage and are sorted unique int64 offsets.
- `src/kohakumoss/adapters/train/lori.py:149-154`: A and alpha are exact clones, while compact B is newly zeroed. Dense calibrated B and optimizer moments cannot enter the initialization artifact.
- `src/kohakumoss/adapters/train/lori.py:180-246`: malformed state, nonfinite values, mixed phases, missing metadata, inconsistent target dimensions/counts, invalid indices and digest mismatches raise errors. No broad fallback hides read failures. `sparse_init` requires zero B; `sparse` permits finite trained values without changing frozen-state digests.
- Native tensor names match the existing `LoRIModule` buffers/parameters and `AdapterSet` raw-state protocol. Metadata supplies the target dimensions that cannot be inferred from a sparse index vector alone. The factory can validate actual targets before creating compact Parameters, and checkpoint saving can retain this contract by updating the phase.
- `scripts/adapters/lori_mask.py:17-22`: the CLI refuses an identical resolved input/output path, preserves other raw metadata, writes the new versioned metadata and immediately validates the saved output. Existing tests exercise CLI tensor/metadata round trips and corruption refusal.
- The helpers and CLI stay within the requested artifact scope. Validation and digest handling have a concrete purpose; no unrelated refactor, extra framework or training mechanism was introduced.

## Issues

### Critical

None found.

### Important

None found.

### Minor

None requiring action for this task.

## Verification and limits

- Reviewed the recorded RED/GREEN, formatting and lint evidence, including **70 passing tests**. Did not repeat that suite.
- Performed one focused, model-free check of a suspected global-quantile size limitation: 33 layers with B `[16384,32]` produced **17,301,504** concatenated scores; `make_sparse_init(state, 0.9)` succeeded in the current environment. That suspicion is not a finding. No model, training or GPU execution occurred.
- Unchanged source was read only for the concrete native-state compatibility question: the LoRI constructor's persistent names and output-size validation, and `AdapterSet.state_dict/load_state_dict/state_metadata/save_state`.
- Cannot independently verify the historical execution of the reported 70 tests from this review alone. Actual MOSS target construction, optimizer/offload ordering, native checkpoint restore and low-VRAM training are not implemented or exercised by this task and receive no runtime certification here.
- The calibration digest is provenance rather than a recomputable proof without the original calibration state. Embedded A/alpha/index digests and count/shape checks remain independently verifiable; the report describes this limit accurately.

## Declined to judge

- Task 3 actual target enumeration, target-set agreement and construction before optimizer/offload/EMA capture: explicitly assigned to the factory integration task; Task 2 provides the required state and shape metadata.
- Task 4 native trained-state metadata retention and checkpoint restore: explicitly assigned to checkpoint integration; the `sparse` phase accepts trained B while retaining the initialization's frozen-state metadata.
- Fresh optimizer moments at training transition: Task 2 transfers no optimizer state; actual optimizer creation belongs to subsequent integration.
- Real-model learning quality, training throughput, peak VRAM and model/data correctness: no training was authorized for this artifact review.

## Assessment

**Spec verdict: Compliant.** The supplied implementation covers the Task 2 artifact interfaces, global mask algorithm, exact projection preservation, B reset, validation, metadata and CLI requirements. No material deviation was found.

**Quality verdict: Approved. Ready to proceed: Yes.** No Critical or Important issue was identified. The metadata/state contract supports the planned factory and checkpoint tasks, whose implementation remains outside this review.
