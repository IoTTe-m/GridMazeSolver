from typing import Sequence, List
from .simulator import ACTION_TO_CHAR, STACK_OP_TO_STR


def genome_to_conditional_pda_string(genome: Sequence[int], num_states: int, num_stack_symbols: int) -> str:
    """Format a conditional PDA genome as a human-readable string."""
    lines: List[str] = []
    per_state_block = 32 * (num_stack_symbols + 1)
    empty_index = num_stack_symbols
    
    # Sensor input labels: (left_free, front_free, right_free)
    sensor_labels = [
        "L=0 F=0 R=0",  # 0b000
        "L=0 F=0 R=1",  # 0b001
        "L=0 F=1 R=0",  # 0b010
        "L=0 F=1 R=1",  # 0b011
        "L=1 F=0 R=0",  # 0b100
        "L=1 F=0 R=1",  # 0b101
        "L=1 F=1 R=0",  # 0b110
        "L=1 F=1 R=1",  # 0b111
    ]

    for state in range(num_states):
        lines.append(f"State {state}:")
        state_base = state * per_state_block

        for top_symbol in range(num_stack_symbols + 1):
            top_name = "EMPTY" if top_symbol == empty_index else str(top_symbol)
            block_base = state_base + top_symbol * 32

            for sensor_idx in range(8):
                gene_base = block_base + sensor_idx * 4
                action = genome[gene_base]
                next_state = genome[gene_base + 1]
                operation = genome[gene_base + 2]
                symbol = genome[gene_base + 3]
                
                action_char = ACTION_TO_CHAR.get(action, '?')
                operation_str = STACK_OP_TO_STR.get(operation, '?')
                lines.append(f"  top={top_name} | {sensor_labels[sensor_idx]} -> (act={action_char}, next={next_state}, op={operation_str}, sym={symbol})")

    return "\n".join(lines)
