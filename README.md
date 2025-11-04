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

Run the trainer to start learning a policy that alternates the green phase intelligently:

```bash
python train_sumo_agent.py --episodes 100 --max-steps 900
```

Key options:

- `--use-gui` launches SUMO with the graphical viewer for live inspection.
- `--eval-every N` periodically pauses training to estimate performance over `--eval-episodes` runs.
- `--save-path PATH` controls where checkpoints are stored (default: `artifacts/dqn_agent.pt`).
- `--no-train` skips learning and only evaluates the current agent (useful for validating saved checkpoints).

The script prints episode rewards and epsilon values so you can monitor convergence. Checkpoints capture both policy and target networks for later reuse.

## Code Layout

```
ReinforcedLearningLabs/
├── agents/
│   ├── base.py         # BaseAgent interface + AgentTransition dataclass
│   └── dqn.py          # PyTorch DQN implementation and configuration dataclass
├── sumo/
│   ├── environment.py  # Gymnasium-compatible SUMOEnvironment and config
│   └── scenario.py     # Scenario builder creating SUMO network/route files
├── train_sumo_agent.py # CLI for training/evaluating agents
├── requirements.txt
└── README.md
```

Both modules follow the DRY principle so shared concerns (e.g., TraCI setup, replay buffer handling) are isolated.

## Extending the Sandbox

- **Custom scenarios**: Derive a new scenario builder that outputs your own network/route files, then pass it to `SUMOEnvironment`.
- **New agents**: Implement additional subclasses of `BaseAgent` (e.g., PPO, actor-critic) and plug them into the trainer.
- **Reward shaping**: Update `_calculate_reward` inside `SUMOEnvironment` to encode metrics such as throughput, average delay, or priority vehicle handling.

Because the environment uses Gymnasium spaces, any off-the-shelf RL library can consume it once wrapped.

## References

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [SUMO Traffic Simulator Tutorial](https://github.com/RoadwayVR/SUMO-Traffic-Simulator-Tutorial)
- [Gymnasium API](https://gymnasium.farama.org/)
