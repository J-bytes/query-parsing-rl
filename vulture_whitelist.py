"""Names vulture cannot see being used. Each entry says who uses it."""
# ruff: noqa: B018
from qprl.dpo.train import DPOModule
from qprl.eval.metrics import EvalReport, FieldScore
from qprl.grpo.algorithm import Advantages
from qprl.grpo.train import GRPOModule
from qprl.prompts import ChatTokenizer
from qprl.sft.train import SFTModule
from qprl.training import GenerationEvalCallback

# Lightning calls these hooks by name.
for _m in (SFTModule, DPOModule, GRPOModule):
    _m.training_step
    _m.configure_optimizers
GRPOModule.automatic_optimization
GenerationEvalCallback.on_train_start
GenerationEvalCallback.on_train_batch_end
GenerationEvalCallback.on_train_end
_.batch_idx  # type: ignore[name-defined]  # Lightning hook signature

# Protocol method parameters.
ChatTokenizer.apply_chat_template
_.conversation  # type: ignore[name-defined]
_.kwargs  # type: ignore[name-defined]

# Dataclass fields serialised with dataclasses.asdict into metrics JSON.
FieldScore.precision
FieldScore.recall
FieldScore.support
EvalReport.micro_precision
EvalReport.micro_recall
EvalReport.micro_f1
EvalReport.mean_example_f1
EvalReport.exact_match
EvalReport.strict_json_rate
EvalReport.lenient_json_rate
EvalReport.schema_valid_rate
Advantages.group_mean  # diagnostics, asserted in tests
Advantages.group_std

# Attributes read by Hugging Face / TRL internals.
_.padding_side  # type: ignore[name-defined]
_.use_cache  # type: ignore[name-defined]
_.current_gradient_accumulation_steps  # type: ignore[name-defined]
