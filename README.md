# SUMO Traffic-Light RL Sandbox

This lab provides a lightweight reinforcement learning workflow for traffic-light control directly on top of **SUMO** and **TraCI**, without the Flow framework. It exposes:

- a reusable `SUMOEnvironment` module for launching SUMO (headless or GUI) and collecting queue statistics
- a modular agent package featuring an abstract `BaseAgent` and a PyTorch-powered `DQNAgent`
- a command-line trainer that ties the pieces together and produces checkpoints ready for further experimentation

The environment generates a self-contained four-way intersection, inspired by the tutorials in the [SUMO Traffic Simulator guide](https://github.com/RoadwayVR/SUMO-Traffic-Simulator-Tutorial), and can be swapped for richer scenarios later on.

## Getting Started

### Prerequisites

1. **Python 3.9+**
2. **SUMO** (provides the `sumo` / `sumo-gui` binaries and the `traci` module)

On Debian/Ubuntu you can install SUMO with:

```bash
sudo apt-get install sumo sumo-tools
```

Make sure `SUMO_HOME` is set and that `$SUMO_HOME/bin` is on your `PATH`.

### Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training a DQN Agent

### Simple Intersection (Default)

Train on a simple four-way intersection:

**Headless (no GUI) - faster training:**
```bash
python train_sumo_agent.py --episodes 100 --max-steps 900
```

**With GUI - visual inspection:**
```bash
python train_sumo_agent.py --episodes 100 --max-steps 900 --use-gui
```

### OSM Map Files

Train on real-world maps from OpenStreetMap:

**Headless training with OSM file:**
```bash
python train_sumo_agent.py --osm-file maps/z9RAW.osm --episodes 100 --max-steps 900
```

**With GUI for visual inspection:**
```bash
python train_sumo_agent.py --osm-file maps/z9RAW.osm --episodes 100 --use-gui
```

**Custom flow rate and traffic light:**
```bash
python train_sumo_agent.py --osm-file maps/z9RAW.osm --flow-rate 800 --tls-id "12345" --episodes 100
```

### Key Options

- `--osm-file PATH` - Path to OSM map file (uses OSM scenario instead of simple intersection)
- `--flow-rate N` - Vehicle flow rate per hour for OSM scenarios (default: 600)
- `--tls-id ID` - Specific traffic light ID to control (defaults to first TLS found)
- `--use-gui` - Launch SUMO with graphical viewer for live inspection (slower but visual)
- `--episodes N` - Number of training episodes
- `--max-steps N` - Maximum steps per episode
- `--eval-every N` - Periodically evaluate performance every N episodes (0 disables)
- `--eval-episodes N` - Number of episodes to average during evaluation (default: 5)
- `--save-path PATH` - Where to save checkpoints (default: `output/run_id/weights.pt`)
- `--output-dir PATH` - Directory to save run outputs (default: `output`)
- `--run-id ID` - Custom run ID (default: auto-generated timestamp)

### Performance Metrics

During training, the script displays:
- **Reward**: Cumulative reward for the episode (higher is better)
- **Epsilon**: Exploration rate (decreases as agent learns)
- **Total Queues**: Total number of vehicles waiting at traffic lights (lower is better)
- **Total Waiting Time**: Sum of waiting times across all lanes in seconds (lower is better)

Example output:
```
Episode 001 | Reward:  -45.23 | Epsilon: 0.950 | Total Queues:  12 | Total Waiting:  45.23s
Episode 002 | Reward:  -38.15 | Epsilon: 0.902 | Total Queues:   8 | Total Waiting:  38.15s
...
```

When `--eval-every` is set, evaluation reports average metrics across multiple episodes for more stable performance estimates.

## Loading and Visualizing Trained Models

After training, you can load a saved model and watch it control traffic lights in real-time using the dedicated visualization script.

### Visualize Trained Agent (GUI)

**Load and visualize a trained model:**
```bash
python visualize_agent.py --load-path output/20251106_192848_00f95735/weights.pt --episodes 5
```

The script automatically detects configuration from the checkpoint directory, so you typically only need to specify the weights path!

**Key options:**
- `--load-path PATH` - **Required.** Path to trained model weights (e.g., `output/run_id/weights.pt`)
- `--osm-file PATH` - OSM map file (auto-detected from checkpoint config if available)
- `--flow-rate N` - Vehicle flow rate (auto-detected from checkpoint config if available)
- `--episodes N` - Number of episodes to run (default: 5)
- `--max-steps N` - Maximum steps per episode (default: 900)
- `--no-gui` - Run without GUI (headless mode, faster but no visualization)
- `--tls-id ID` - Specific traffic light ID (for OSM scenarios)

**Example workflow:**
```bash
# 1. Train a model (saves to output/run_id/weights.pt)
python train_sumo_agent.py --osm-file maps/sanFrancisco.osm --flow-rate 800 --episodes 100

# 2. Visualize the trained model (auto-detects config from checkpoint)
python visualize_agent.py --load-path output/20251106_192848_00f95735/weights.pt --episodes 5

# 3. Or specify options explicitly
python visualize_agent.py \
    --load-path output/20251106_192848_00f95735/weights.pt \
    --osm-file maps/sanFrancisco.osm \
    --flow-rate 800 \
    --episodes 5
```

**Note:** The GUI will open automatically. Watch how the trained agent controls traffic lights to minimize queues and waiting times. Close the SUMO GUI window when done.

### Evaluate Without GUI

**Headless evaluation (faster, no visualization):**
```bash
python visualize_agent.py --load-path output/20251106_192848_00f95735/weights.pt --episodes 10 --no-gui
```

## Analyzing Training Results

Use the provided Jupyter notebook to analyze training metrics:

```bash
jupyter notebook analyze_training.ipynb
```

The notebook automatically loads the most recent training run and provides:
- Training progress visualizations (reward, queues, waiting times)
- Loss curves and action distribution analysis
- Per-lane performance breakdowns
- Comparison tools for multiple training runs

Metrics are automatically saved to `output/run_id/metrics.json` during training.

## Code Layout

```
ReinforcedLearningLabs/
├── agents/
│   ├── base.py         # BaseAgent interface + AgentTransition dataclass
│   └── dqn.py          # PyTorch DQN implementation and configuration dataclass
├── sumo/
│   ├── environment.py  # Gymnasium-compatible SUMOEnvironment and config
│   └── scenario.py     # Scenario builders (SimpleIntersectionScenario, OSMScenario)
├── maps/
│   └── *.osm           # OpenStreetMap files for real-world scenarios
├── output/              # Training outputs (weights, metrics, configs)
│   └── run_id/         # Each training run gets a unique directory
│       ├── weights.pt   # Trained model weights
│       ├── metrics.json # Training metrics and episode data
│       └── config.json  # Training configuration
├── train_sumo_agent.py # CLI for training agents
├── visualize_agent.py  # CLI for visualizing trained agents in SUMO GUI
├── analyze_training.ipynb # Jupyter notebook for analyzing training results
├── requirements.txt
└── README.md
```

Both modules follow the DRY principle so shared concerns (e.g., TraCI setup, replay buffer handling) are isolated.

## Scenarios

The framework supports two types of scenarios:

1. **SimpleIntersectionScenario**: A minimal four-way intersection with a single traffic light (default)
2. **OSMScenario**: Real-world maps from OpenStreetMap with automatic traffic light detection

OSM scenarios automatically:
- Convert OSM files to SUMO networks using `netconvert`
- Detect and handle roundabouts
- Identify traffic lights from OSM data
- Generate realistic vehicle routes using `randomTrips.py` and `duarouter`
- Clean up network geometry and join nearby junctions

## Extending the Sandbox

- **Custom scenarios**: Derive a new scenario builder that outputs your own network/route files, then pass it to `SUMOEnvironment`. See `OSMScenario` for an example.
- **New agents**: Implement additional subclasses of `BaseAgent` (e.g., PPO, actor-critic) and plug them into the trainer.
- **Reward shaping**: Update `_calculate_reward` inside `SUMOEnvironment` to encode metrics such as throughput, average delay, or priority vehicle handling.
- **OSM map preparation**: Use `netconvert` with custom options via `OSMScenario(netconvert_options=[...])` for advanced network processing.

Because the environment uses Gymnasium spaces, any off-the-shelf RL library can consume it once wrapped.

## References

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [SUMO Traffic Simulator Tutorial](https://github.com/RoadwayVR/SUMO-Traffic-Simulator-Tutorial)
- [Gymnasium API](https://gymnasium.farama.org/)
