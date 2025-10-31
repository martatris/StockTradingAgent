# Lagrangian-Constrained Stock Trading Agent (Reinforcement Learning)

This project implements a **multi-asset portfolio trading agent** using **Reinforcement Learning (RL)** with a **Lagrangian constraint** to limit portfolio drawdown.  
The model uses **Proximal Policy Optimization (PPO)** combined with a **dual variable update** that enforces a *maximum drawdown constraint* — balancing risk and reward.

---

## Features

- **Automatic stock data download** (AAPL, MSFT, GOOG, AMZN) using `yfinance`
- **Multi-asset continuous portfolio allocation** (weights sum to 1)
- **Custom OpenAI Gym environment**
- **Drawdown-constrained PPO training**
- **Lagrangian dual updates** (adaptive risk control)
- **PyTorch-based actor–critic architecture**
- **Easy to modify and extend**

---

## Project Structure

```
.
├── train_lagrangian_portfolio.py   # Main training script
├── data/
│   └── multi_asset.csv             # Auto-generated dataset (if not found)
└── README.md
```

---

## Installation

1. **Create environment**
   ```bash
   python -m venv rl-env
   source rl-env/bin/activate  # (Windows: rl-env\Scripts\activate)
   ```

2. **Install dependencies**
   ```bash
   pip install torch numpy pandas yfinance gym tqdm matplotlib
   ```

---

## Running the Training

Simply run:
```bash
python train_lagrangian_portfolio.py
```

- The script will automatically download historical stock data from Yahoo Finance.  
- It will train the RL agent over multiple epochs.  
- You’ll see logs like:

  ```
  [Update 0048] steps=98304 rollout_return=0.939416 max_dd=0.7246 lambda=0.280376
  ```

### Example Meaning:
| Term | Description |
|------|--------------|
| `steps` | Total environment steps taken |
| `rollout_return` | Average return over the last rollout |
| `max_dd` | Maximum drawdown observed (risk metric) |
| `lambda` | Lagrange multiplier (penalty for risk constraint violation) |

---

## Algorithm Overview

The agent optimizes the objective:

\[
\max_\pi \; \mathbb{E}[R(\pi)] - \lambda (\text{Drawdown} - \text{Limit})
\]

where:
- \( R(\pi) \): expected portfolio return  
- \( \text{Drawdown} \): maximum drop from peak portfolio value  
- \( \lambda \): dual variable (Lagrange multiplier), updated automatically each iteration  

This ensures the policy improves performance **without exceeding the drawdown constraint**.

---

## Configuration

You can adjust hyperparameters inside the script:

```python
trainer.train(epochs=20, steps_per_epoch=300)
```

| Parameter | Meaning |
|------------|----------|
| `epochs` | Number of training epochs |
| `steps_per_epoch` | Steps per epoch |
| `drawdown_limit` | Maximum allowed drawdown (e.g. 0.2 = 20%) |
| `lam_lr` | Learning rate for Lagrange multiplier |

---

## Tips

- Use `CTRL + C` to stop training early.
- To change tickers, edit:
  ```python
  tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
  ```
- All downloaded data is saved in `data/multi_asset.csv`.

---

## Expected Behavior

During training, the agent should:
- Gradually increase average returns (`rollout_return`)
- Reduce `max_dd` as the drawdown penalty (`λ`) rises
- Eventually stabilize around a steady λ value

---

## Future Extensions

- Add **GAE (Generalized Advantage Estimation)**  
- Support **minibatch PPO updates**  
- Integrate **TensorBoard logging**  
- Extend to **crypto, forex, or futures** data  
- Deploy trained policy for live paper trading

---

## 🧾 Citation / Credit

Inspired by:
- *Achiam et al. (2017), “Constrained Policy Optimization”*  
- *Schulman et al. (2017), “Proximal Policy Optimization Algorithms”*  

Developed and tested with Python 3.9+, PyTorch ≥1.9.
