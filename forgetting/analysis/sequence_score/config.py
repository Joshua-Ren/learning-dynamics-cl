from dataclasses import asdict, dataclass
from typing import Literal


GradientConvention = Literal["logprob", "nll"]
TokenReduction = Literal["sum", "mean"]
ObservationObjective = Literal["uniform_options", "ifmass"]
UpdateTokenMode = Literal["all_supervised", "first_final_answer_token", "first_response_token"]


@dataclass(frozen=True)
class SequenceScoreConfig:
    gradient_convention: GradientConvention = "logprob"
    primary_token_reduction: TokenReduction = "mean"
    kembd_mode: Literal["prefix", "full_sequence_scalar"] = "prefix"
    hidden_stream: Literal["attn_input_rmsnorm"] = "attn_input_rmsnorm"
    exclude_special_tokens_from_overlap: bool = False
    update_token_mode: UpdateTokenMode = "all_supervised"
    final_answer_markers: tuple[str, ...] = ("####", "Answer:")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PairScoreResult:
    objective: ObservationObjective
    ch1_sum: float
    ch2_sum: float
    ch2_hidden_sum: float
    ch2_embd_sum: float
    total_sum: float
    ch1_mean: float
    ch2_mean: float
    ch2_hidden_mean: float
    ch2_embd_mean: float
    total_mean: float
    num_update_tokens: int
    mean_kembd: float
    max_kembd: float

    def to_dict(self) -> dict:
        return asdict(self)
