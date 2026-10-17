# WALL-E and EVE's Signal

A pure-Python simulation of WALL-E searching a grid for EVE using signal scans, clues, belief updates, battery management, and recharge decisions.

## Run

```bash
python walle_simulation.py
```

The script creates `config.json` on first run, prints the action trace and ASCII map, and writes the complete trace to `action_trace.json`.

## Requirements

- Python 3.8 or newer
- Standard library only

The default mission searches a 10x10 grid, collects two clues, and reports whether EVE was found before the battery or time budget expires.
