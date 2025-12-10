from typing import Sequence, List
from .simulator import ACTION_TO_CHAR, STACK_OP_TO_STR

def genome_to_conditional_pda_string(genome: Sequence[int], n_states: int, n_stack_syms: int) -> str:
    """Pretty print the conditional PDA genome: for each state and top-symbol list transitions."""
    lines: List[str] = []
    per_state_block = 8 * (n_stack_syms + 1)
    EMPTY_IDX = n_stack_syms
    for s in range(n_states):
        lines.append(f"State {s}:")
        state_base = s * per_state_block
        for ts in range(n_stack_syms + 1):
            ts_name = f"{ts}" if ts != EMPTY_IDX else "EMPTY"
            block_base = state_base + ts * 8
            blocked = genome[block_base:block_base + 4]
            free = genome[block_base + 4:block_base + 8]
            a_b, n_b, op_b, sym_b = blocked
            a_f, n_f, op_f, sym_f = free
            lines.append(
                f"  top={ts_name} | BLOCKED -> (act={ACTION_TO_CHAR.get(a_b,'?')}, next={n_b}, op={STACK_OP_TO_STR.get(op_b,'?')}, sym={sym_b})"
            )
            lines.append(
                f"  top={ts_name} | FREE    -> (act={ACTION_TO_CHAR.get(a_f,'?')}, next={n_f}, op={STACK_OP_TO_STR.get(op_f,'?')}, sym={sym_f})"
            )
    return "\n".join(lines)
