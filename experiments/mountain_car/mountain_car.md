# MountainCar Discrete Pathologies

## 1. Q-Learning

### Good Learning (Standard)
*Uses Optimistic Initialization (0.0) to force exploration, but decays epsilon to 0.0 so the policy perfectly locks in once the goal is found.*
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model q_learning \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon 1.0 \
  --epsilon_min 0.0 \
  --epsilon_decay 0.9998 \
  --q_init_val 0.0 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/q_learning/standard
```

### Perpetual Reshaping
*Keeps `epsilon_min` at 0.05. The constant 5% random actions knock the agent off its path into optimistic 0.0 states, causing the Q-table to violently rewrite itself forever.*
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model q_learning \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon 1.0 \
  --epsilon_min 0.05 \
  --epsilon_decay 0.999 \
  --q_init_val 0.0 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/q_learning/perpetual_reshaping
```

### Exploration Deprivation (Limited Exploration)
*Uses Pessimistic Initialization (-200.0). Because every new state gives -200, the agent is terrified of unexplored states and cowers in the valley floor, completely failing to explore.*
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model q_learning \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon 0.01 \
  --epsilon_min 0.01 \
  --epsilon_decay 0.0 \
  --q_init_val 0.0 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/q_learning/limited_exploration
```

### Catastrophic Forgetting
*Uses a huge learning rate (`0.5`). When the agent accidentally explores a bad state, it violently overwrites its Q-table instantly, forgetting everything it learned.*
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model q_learning \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --alpha 0.5 \
  --gamma 0.99 \
  --epsilon 1.0 \
  --epsilon_min 0.01 \
  --epsilon_decay 0.9998 \
  --q_init_val 0.0 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/q_learning/catastrophic_forgetting
```

---

## 2. SARSA

### Good Learning (Standard)
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model sarsa \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon 1.0 \
  --epsilon_min 0.0 \
  --epsilon_decay 0.9998 \
  --q_init_val 0.0 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/sarsa/standard
```

### Perpetual Reshaping
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model sarsa \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon 1.0 \
  --epsilon_min 0.05 \
  --epsilon_decay 0.999 \
  --q_init_val 0.0 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/sarsa/perpetual_reshaping
```

### Exploration Deprivation
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model sarsa \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon 0.01 \
  --epsilon_min 0.01 \
  --epsilon_decay 0.0 \
  --q_init_val 0.0 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/sarsa/limited_exploration
```

### Catastrophic Forgetting
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model sarsa \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --alpha 0.5 \
  --gamma 0.99 \
  --epsilon 1.0 \
  --epsilon_min 0.01 \
  --epsilon_decay 0.9998 \
  --q_init_val 0.0 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/sarsa/catastrophic_forgetting
```

---

## 3. DQN

### Good Learning (Standard)
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model dqn \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --lr 0.0001 \
  --gamma 0.99 \
  --epsilon 1.0 \
  --epsilon_min 0.0 \
  --epsilon_decay_steps 500000 \
  --memory_size 200000 \
  --batch_size 256 \
  --replay_each 4 \
  --target_update_every 2000 \
  --hidden_layers 64 64 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/dqn/standard
```

### Perpetual Reshaping
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model dqn \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --lr 0.001 \
  --gamma 0.99 \
  --epsilon 1.0 \
  --epsilon_min 0.05 \
  --epsilon_decay 0.999 \
  --memory_size 10000 \
  --batch_size 64 \
  --target_update_every 1 \
  --hidden_layers 64 64 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/dqn/perpetual_reshaping
```

### Exploration Deprivation
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model dqn \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --lr 0.0001 \
  --gamma 0.99 \
  --epsilon 0.01 \
  --epsilon_min 0.01 \
  --epsilon_decay_steps 1 \
  --memory_size 200000 \
  --batch_size 256 \
  --replay_each 4 \
  --target_update_every 2000 \
  --hidden_layers 64 64 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/dqn/limited_exploration
```

### Catastrophic Forgetting
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model dqn \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --lr 0.01 \
  --gamma 0.99 \
  --epsilon 1.0 \
  --epsilon_min 0.01 \
  --epsilon_decay 0.9998 \
  --memory_size 64 \
  --batch_size 64 \
  --target_update_every 100 \
  --hidden_layers 64 64 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/dqn/catastrophic_forgetting
```

---

## 4. PPO

### Good Learning (Standard)
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model ppo \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --lr 0.0003 \
  --gamma 0.99 \
  --clip_ratio 0.2 \
  --lam 0.95 \
  --entropy_coef 0.01 \
  --entropy_min 0.0 \
  --entropy_decay_frac 0.5 \
  --value_coef 0.5 \
  --train_iters 10 \
  --batch_size 2048 \
  --hidden_layers 64 64 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/ppo/standard
```

### Perpetual Reshaping
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model ppo \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --lr 0.0003 \
  --gamma 0.99 \
  --clip_ratio 0.2 \
  --lam 0.95 \
  --entropy_coef 0.1 \
  --entropy_min 0.05 \
  --entropy_decay_frac 0.5 \
  --value_coef 0.5 \
  --train_iters 10 \
  --batch_size 2048 \
  --hidden_layers 64 64 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/ppo/perpetual_reshaping
```

### Exploration Deprivation
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model ppo \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --lr 0.0003 \
  --gamma 0.99 \
  --clip_ratio 0.2 \
  --lam 0.95 \
  --entropy_coef 0.0 \
  --entropy_min 0.0 \
  --entropy_decay_frac 0.0 \
  --value_coef 0.5 \
  --train_iters 10 \
  --batch_size 128 \
  --hidden_layers 64 64 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/ppo/limited_exploration
```

### Catastrophic Forgetting
```bash
/data/ondra/venvs/atat-env/bin/python3 -m experiments.mountain_car.mountain_car_discrete \
  --model ppo \
  --seed 42 \
  --episodes 30000 \
  --evaluate_each 1000 \
  --evaluate_for 500 \
  --lr 0.0003 \
  --gamma 0.99 \
  --clip_ratio 0.2 \
  --lam 0.95 \
  --entropy_coef 0.01 \
  --entropy_min 0.0 \
  --entropy_decay_frac 0.5 \
  --value_coef 0.5 \
  --train_iters 50 \
  --batch_size 1024 \
  --hidden_layers 64 64 \
  --log_dir /datafast/ondra/trajectories/behavioral_ontogeny/mountain_car/ppo/catastrophic_forgetting
```
