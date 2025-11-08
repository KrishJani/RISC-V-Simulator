# RISC-V Single-Stage Processor Simulator

A RISC-V RV32I single-stage processor simulator implemented in Python. This project simulates a processor that executes RISC-V assembly instructions in a single cycle.

## Features

- **Instruction Support**: Implements RISC-V RV32I instruction set including:
  - R-type instructions (ADD, SUB, XOR, OR, AND, SLL, SRL, SRA, SLT, SLTU)
  - I-type instructions (ADDI, XORI, ORI, ANDI, LW, LH, LB, LBU, LHU, JALR)
  - S-type instructions (SW, SH, SB)
  - SB-type branch instructions (BEQ, BNE, BLT, BGE, BLTU, BGEU)
  - U-type instructions (LUI, AUIPC)
  - UJ-type instructions (JAL)
  - HALT instruction

- **Components**:
  - Instruction Memory (InsMem): Reads and decodes RISC-V instructions
  - Data Memory (DataMem): Handles data storage and retrieval
  - Register File: Manages 32 general-purpose registers (x0-x31)
  - Single-Stage Core: Executes instructions in a single cycle

## Project Structure

```
phase1/
├── code/
│   ├── main.py              # Main simulator code
│   └── requirements.txt     # Python dependencies
└── submissions/
    └── Report.pdf          # Project report and documentation
```

## Requirements

- Python 3.x
- argparse (built-in Python module)

## Usage

Run the simulator with a test case:

```bash
cd code
python3 main.py --iodir testcase0
```

Replace `testcase0` with `testcase1` or `testcase2` to run other test cases.

## Input Files

Each test case directory should contain:
- `imem.txt`: Instruction memory (binary instructions, byte-addressable)
- `dmem.txt`: Initial data memory state (binary data, byte-addressable)

## Output Files

After execution, the simulator generates:
- `SS_RFResult.txt`: Register file state after each cycle
- `SS_DMEMResult.txt`: Final data memory state
- `StateResult_SS.txt`: Processor state (PC, nop flag) after each cycle
- `PerformanceMetrics.txt`: Performance statistics (cycles, instructions, CPI, IPC)

## Example

```bash
python3 code/main.py --iodir code/testcase0
```

## License

This project is part of a Computer Systems Architecture course assignment.

