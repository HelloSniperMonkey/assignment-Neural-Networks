"""Builds assignment3.ipynb from the cells below (run: .venv/bin/python build_notebook.py)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s, tags=(): cells.append(nbf.v4.new_code_cell(s.strip(), metadata={"tags": list(tags)}))

md(r"""
# Assignment 3: Neural Networks — Feedforward MLP from Scratch on CIFAR-10
**Course:** CS4101 | CS6103

| Name | Roll number |
|---|---|
| NAME_1 | ROLL_1 |
| NAME_2 | ROLL_2 |
| NAME_3 | ROLL_3 |

**Weights & Biases project:** WANDB_LINK

### Execution instructions
* **Dependencies:** Python 3.12, `numpy`, `keras` (only to load CIFAR-10; backend `jax`), `wandb`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
  ```
  pip install numpy keras jax wandb pandas matplotlib seaborn scikit-learn jupyter
  wandb login            # once, to sync runs to the online project
  ```
* Run all cells top to bottom (`Kernel → Restart & Run All`). Runs take about 1–2 minutes on a laptop CPU.
* Set `WANDB_MODE=offline` in the environment to run without a W&B account. Offline runs can be uploaded later with `wandb sync wandb/offline-run-*`.

### Random seeds
| Purpose | Seed |
|---|---|
| Train/validation subset selection | `np.random.default_rng(42)` (as specified) |
| Weight initialisation | `np.random.default_rng(INIT_SEED = 0)` |
| Per-epoch shuffling | `np.random.default_rng(SHUFFLE_SEED = 1)` (one generator per run, so every run sees the same batch order) |
""")

code(r"""
import os, time, json, copy
os.environ.setdefault("KERAS_BACKEND", "jax")   # keras is used ONLY to load the dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix, classification_report
from keras.datasets import cifar10

DATA_SEED, INIT_SEED, SHUFFLE_SEED = 42, 0, 1
WANDB_PROJECT = "cifar10-mlp-from-scratch"
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
CLASS_NAMES = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
               "Dog", "Frog", "Horse", "Ship", "Truck"]
np.set_printoptions(precision=4, suppress=True)
print("numpy", np.__version__, "| wandb", wandb.__version__, "| wandb mode:", WANDB_MODE)
""")

md(r"""
## Dataset and preprocessing
1. Sample 1,000 images per class with `rng = np.random.default_rng(42)`, going through the classes in order 0–9 without resetting the generator. The first 800 of each class go to training and the remaining 200 to validation. This gives 8,000 training and 2,000 validation images.
2. Scale pixels to `[0, 1]` by dividing by 255. No other normalisation is applied.
3. Flatten each image (all three colour channels) into a 3,072-dimensional vector. Labels stay as integers, and one-hot vectors are built where the loss needs them.
4. Keep the 10,000 official test images untouched until Question 4.
""")

code(r"""
(x_train_full, y_train_full), (x_test_raw, y_test_raw) = cifar10.load_data()
y_train_full, y_test_raw = y_train_full.ravel(), y_test_raw.ravel()

rng = np.random.default_rng(DATA_SEED)
train_idx, val_idx = [], []
for c in range(10):                                         # classes in label order
    class_indices = np.where(y_train_full == c)[0]          # ascending original indices
    chosen = rng.choice(class_indices, size=1000, replace=False)
    train_idx.append(chosen[:800])
    val_idx.append(chosen[800:])
train_idx, val_idx = np.concatenate(train_idx), np.concatenate(val_idx)

def preprocess(x):
    # float, scale to [0, 1], flatten 32x32x3 -> 3072 (channels retained)
    return (x.astype(np.float64) / 255.0).reshape(len(x), -1)

X_train, y_train = preprocess(x_train_full[train_idx]), y_train_full[train_idx]
X_val,   y_val   = preprocess(x_train_full[val_idx]),   y_train_full[val_idx]
X_test,  y_test  = preprocess(x_test_raw),              y_test_raw

def one_hot(y, k=10):
    return np.eye(k)[y]

print("train", X_train.shape, "val", X_val.shape, "test", X_test.shape)
print("pixel range", X_train.min(), X_train.max())
print("train per class", np.bincount(y_train), "\nval per class  ", np.bincount(y_val))
assert len(np.intersect1d(train_idx, val_idx)) == 0
""")

md(r"""
## Question 1: Forward propagation

**Architecture.** The input has 3,072 features, followed by one or two fully connected hidden layers of configurable width with **sigmoid** or **ReLU** activation, and a 10-unit output layer with softmax.

**Numerically stable softmax.** $\mathrm{softmax}(z)_c = e^{z_c - \max_j z_j} / \sum_k e^{z_k - \max_j z_j}$. Subtracting the row maximum leaves the result unchanged and prevents `exp` from overflowing.
Sigmoid is evaluated as $1/(1+e^{-z})$ with $z$ clipped to $[-500, 500]$, which also avoids overflow.

**Parameter initialisation (documented choice).** Weights are drawn from a zero-mean normal distribution, using the seeded generator `default_rng(INIT_SEED=0)`, and all biases start at zero.
* **He/Kaiming normal** for ReLU hidden layers: $\sigma = \sqrt{2/n_\text{in}}$. ReLU zeroes about half of its inputs, and the factor 2 keeps the activation variance roughly constant from layer to layer.
* **Xavier/Glorot normal** for sigmoid hidden layers and for the softmax output layer: $\sigma = \sqrt{2/(n_\text{in}+n_\text{out})}$. This scale suits activations that are approximately linear around 0.

The weight matrix of layer $l$ has shape $(n_{l-1}, n_l)$ and the bias has shape $(n_l,)$. A batch $X \in \mathbb{R}^{B\times 3072}$ therefore propagates as $Z_l = A_{l-1}W_l + b_l$.
""")

code(r"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(z, 0.0)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)         # stability shift
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

ACTIVATIONS = {"sigmoid": sigmoid, "relu": relu}


class MLP:
    # Fully connected network: 3072 -> hidden... -> 10 (softmax).
    # params: {"W1","b1",...,"WL","bL"}; W_l has shape (n_{l-1}, n_l), b_l has shape (n_l,).

    def __init__(self, hidden_sizes=(64,), activation="relu", n_in=3072, n_out=10, seed=INIT_SEED):
        assert 1 <= len(hidden_sizes) <= 2, "one or two hidden layers supported"
        assert activation in ACTIVATIONS
        self.hidden_sizes, self.activation = tuple(hidden_sizes), activation
        self.sizes = [n_in, *hidden_sizes, n_out]
        self.L = len(self.sizes) - 1
        rng = np.random.default_rng(seed)
        self.params = {}
        for l in range(1, self.L + 1):
            fan_in, fan_out = self.sizes[l - 1], self.sizes[l]
            if l < self.L and activation == "relu":
                std = np.sqrt(2.0 / fan_in)                 # He
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))     # Xavier/Glorot
            self.params[f"W{l}"] = rng.normal(0.0, std, size=(fan_in, fan_out))
            self.params[f"b{l}"] = np.zeros(fan_out)

    def n_params(self):
        return sum(p.size for p in self.params.values())

    def forward(self, X):
        # Returns (probabilities (B, 10), cache needed by backward).
        act = ACTIVATIONS[self.activation]
        cache = {"A0": X}
        A = X
        for l in range(1, self.L + 1):
            Z = A @ self.params[f"W{l}"] + self.params[f"b{l}"]
            A = act(Z) if l < self.L else softmax(Z)
            cache[f"Z{l}"], cache[f"A{l}"] = Z, A
        return A, cache

    def predict_proba(self, X):
        return self.forward(X)[0]
""")

md(r"""
### Q1 verification: output shape, valid probabilities, parameter shapes and count
For **3072 → 64 → 10**:

| Parameter | Shape | Count |
|---|---|---|
| $W_1$ | (3072, 64) | 196,608 |
| $b_1$ | (64,) | 64 |
| $W_2$ | (64, 10) | 640 |
| $b_2$ | (10,) | 10 |
| **Total** | | **197,322** |

$(3072\cdot 64 + 64) + (64\cdot 10 + 10) = 196{,}672 + 650 = 197{,}322$ trainable parameters.
""")

code(r"""
for act in ["sigmoid", "relu"]:
    net = MLP(hidden_sizes=(64,), activation=act)
    P = net.predict_proba(X_train[:16])
    print(f"[{act}] output shape {P.shape}; all in [0,1]: {bool(((P >= 0) & (P <= 1)).all())}; "
          f"row sums ~ 1: {np.allclose(P.sum(axis=1), 1.0)}")
    assert P.shape == (16, 10) and np.allclose(P.sum(axis=1), 1.0) and (P >= 0).all()

net = MLP(hidden_sizes=(64,), activation="relu")
for k, v in net.params.items():
    print(f"{k}: shape {v.shape}, {v.size:,} params")
print(f"Total trainable parameters (3072 -> 64 -> 10): {net.n_params():,}")
assert net.n_params() == 3072 * 64 + 64 + 64 * 10 + 10 == 197_322

# Two-hidden-layer configuration is supported too
net2 = MLP(hidden_sizes=(64, 64), activation="sigmoid")
print("Two hidden layers 3072 -> 64 -> 64 -> 10:", net2.predict_proba(X_train[:4]).shape, f"{net2.n_params():,} params")

# Stability check: huge logits must not overflow
print("softmax of extreme logits:", softmax(np.array([[1000.0, 1000.0, -1000.0]])))
""")

md(r"""
## Question 2: Backpropagation and training

**Loss (mean categorical cross-entropy):**
$\mathcal{L} = -\frac{1}{B}\sum_{i}\sum_{c} y_{ic}\log\hat p_{ic}$.
For numerical stability the loss is computed from the **logits** with log-softmax, $\log\hat p = z - \max z - \log\sum e^{z-\max z}$, rather than as `log(softmax(z))`. Taking the log of a softmax output can produce `log(0) = -inf`.

**(a) Gradients.** With $\delta_L = \partial\mathcal{L}/\partial Z_L$:
* Softmax combined with cross-entropy gives $\delta_L = (\hat P - Y)/B$.
* For each layer $l$: $\nabla W_l = A_{l-1}^\top \delta_l$, $\nabla b_l = \sum_i \delta_{l,i}$, and $\delta_{l-1} = (\delta_l W_l^\top)\odot f'(Z_{l-1})$.
* The activation derivatives are $\sigma'(z) = \sigma(z)(1-\sigma(z))$ and $\mathrm{ReLU}'(z) = \mathbb{1}[z>0]$.

The analytic gradients are checked against central finite differences below.
""")

code(r"""
def cross_entropy(probs_or_logits, y, from_logits=False):
    # Mean categorical cross-entropy. y: integer labels (one-hot built internally).
    Y = one_hot(y, probs_or_logits.shape[1])
    if from_logits:
        logp = log_softmax(probs_or_logits)
    else:
        logp = np.log(np.clip(probs_or_logits, 1e-12, 1.0))   # guard against log(0)
    return -(Y * logp).sum(axis=1).mean()

def backward(model, cache, y):
    # Gradients of the mean CE loss w.r.t. every W_l and b_l.
    B = len(y)
    grads = {}
    delta = (cache[f"A{model.L}"] - one_hot(y)) / B             # dL/dZ_L (softmax + CE)
    for l in range(model.L, 0, -1):
        A_prev = cache[f"A{l-1}"]
        grads[f"W{l}"] = A_prev.T @ delta
        grads[f"b{l}"] = delta.sum(axis=0)
        if l > 1:
            dA = delta @ model.params[f"W{l}"].T
            if model.activation == "sigmoid":
                delta = dA * A_prev * (1.0 - A_prev)               # A_prev = sigmoid(Z_{l-1})
            else:
                delta = dA * (cache[f"Z{l-1}"] > 0)
    return grads

def loss_of(model, X, y):
    _, cache = model.forward(X)
    return cross_entropy(cache[f"Z{model.L}"], y, from_logits=True)
""")

code(r"""
# Gradient check: analytic vs. central finite differences on a small network
def grad_check(activation, hidden, n_checks=20, eps=1e-6, seed=123):
    r = np.random.default_rng(seed)
    m = MLP(hidden_sizes=hidden, activation=activation, n_in=20, seed=seed)
    X, y = r.normal(size=(7, 20)), r.integers(0, 10, size=7)
    _, cache = m.forward(X)
    g = backward(m, cache, y)
    worst = 0.0
    for name, p in m.params.items():
        for _ in range(n_checks):
            i = tuple(r.integers(0, s) for s in p.shape)
            old = p[i]
            p[i] = old + eps; lp = loss_of(m, X, y)
            p[i] = old - eps; lm = loss_of(m, X, y)
            p[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[name][i]) / max(1e-8, abs(num) + abs(g[name][i])))
    return worst

for act in ["sigmoid", "relu"]:
    for hidden in [(8,), (8, 6)]:
        err = grad_check(act, hidden)
        print(f"{act:7s} hidden={hidden}: max relative error = {err:.2e}")
        assert err < 1e-5
""")

md(r"""
**(b) Optimisers.**
* Mini-batch SGD: $\theta_t = \theta_{t-1} - \eta\nabla_\theta\mathcal{L}_{\mathcal{B}_t}(\theta_{t-1})$.
* Momentum: $v_t = 0.9\,v_{t-1} + \nabla_\theta\mathcal{L}_{\mathcal{B}_t}(\theta_{t-1})$, then $\theta_t = \theta_{t-1} - \eta v_t$. Every velocity starts at zero when a run begins and is kept across batches and epochs.

**(c)** Before every epoch the training set is shuffled with a per-run generator `default_rng(SHUFFLE_SEED=1)`. Weights are initialised with `default_rng(INIT_SEED=0)`.

**(d)** After every epoch the model is evaluated on the **complete** training subset (8,000 images) and validation subset (2,000 images), and loss and accuracy are recorded for both.

**Checkpointing.** A run's checkpoint is its parameters at the first epoch that reaches the run's highest validation accuracy. Only a *strictly* higher accuracy replaces the stored copy, so the earliest checkpoint is the one kept.

**Training time.** Measured with `time.perf_counter()` and summed over the mini-batch steps only (forward, backward and parameter update). The per-epoch full-set evaluations, checkpoint copies and W&B logging are excluded, so the number reflects the cost of training itself.

Everything is driven by a `config` dict, so experiments never require changes to the training loop.
""")

code(r"""
class SGD:
    def __init__(self, lr):
        self.lr = lr
    def step(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]

class Momentum:
    def __init__(self, lr, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None
    def step(self, params, grads):
        if self.v is None:                                    # zero velocity at start of run
            self.v = {k: np.zeros_like(p) for k, p in params.items()}
        for k in params:
            self.v[k] = self.beta * self.v[k] + grads[k]      # v_t = 0.9 v_{t-1} + grad
            params[k] -= self.lr * self.v[k]                  # theta_t = theta_{t-1} - lr v_t

OPTIMIZERS = {"sgd": SGD, "momentum": Momentum}

def evaluate(model, X, y, batch_size=2000):
    # Loss and accuracy on a full subset (batched only to bound memory).
    losses, correct = [], 0
    for i in range(0, len(X), batch_size):
        _, cache = model.forward(X[i:i + batch_size])
        logits = cache[f"Z{model.L}"]
        losses.append(cross_entropy(logits, y[i:i + batch_size], from_logits=True) * len(logits))
        correct += (logits.argmax(axis=1) == y[i:i + batch_size]).sum()
    return sum(losses) / len(X), correct / len(X)

def train(config, X_tr, y_tr, X_va, y_va, use_wandb=True, verbose=True):
    # Train one configuration. Returns a dict with history, best checkpoint and timing.
    model = MLP(config["hidden_sizes"], config["activation"], seed=config["init_seed"])
    opt = OPTIMIZERS[config["optimizer"]](config["lr"])
    shuffle_rng = np.random.default_rng(config["shuffle_seed"])
    bs = config["batch_size"]

    run = None
    if use_wandb:
        run = wandb.init(project=WANDB_PROJECT, name=config["name"], config={**config, "n_params": model.n_params()},
                         mode=WANDB_MODE, reinit="finish_previous")

    history = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]}
    best = {"val_acc": -1.0, "epoch": None, "params": None}
    train_time = 0.0
    for epoch in range(1, config["epochs"] + 1):
        perm = shuffle_rng.permutation(len(X_tr))             # shuffle before every epoch
        t0 = time.perf_counter()
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i + bs]
            _, cache = model.forward(X_tr[idx])
            grads = backward(model, cache, y_tr[idx])
            opt.step(model.params, grads)
        train_time += time.perf_counter() - t0

        tr_loss, tr_acc = evaluate(model, X_tr, y_tr)
        va_loss, va_acc = evaluate(model, X_va, y_va)
        for k, v in zip(history, [epoch, tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(v)
        if va_acc > best["val_acc"]:                          # strict '>' keeps the earliest epoch
            best = {"val_acc": va_acc, "epoch": epoch, "params": copy.deepcopy(model.params)}
        if run:
            run.log({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                     "val_loss": va_loss, "val_acc": va_acc, "train_time_s": train_time})
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f}")

    result = {"config": config, "n_params": model.n_params(), "history": history,
              "best_val_acc": best["val_acc"], "best_epoch": best["epoch"],
              "best_params": best["params"], "train_time_s": train_time}
    if run:
        run.summary.update({"best_val_acc": best["val_acc"], "best_epoch": best["epoch"],
                            "n_params": model.n_params(), "train_time_s": train_time})
        run.finish()
    return result

def make_config(name, hidden_sizes, activation, optimizer="momentum", lr=0.005, batch_size=64, epochs=40):
    return dict(name=name, hidden_sizes=list(hidden_sizes), activation=activation, optimizer=optimizer,
                lr=lr, batch_size=batch_size, epochs=epochs, init_seed=INIT_SEED, shuffle_seed=SHUFFLE_SEED)
""")

md(r"""
### (e) A complete training run
**Chosen setup:** architecture 3072 → 64 → 10 with ReLU, momentum optimiser ($\beta = 0.9$), learning rate $\eta = 0.005$, batch size 64, 40 epochs.

The learning rate was picked in a short preliminary sweep over {0.01, 0.005, 0.001} that looked at validation accuracy only. At 0.01 the ReLU validation loss was noisy, and at 0.001 the sigmoid networks learned too slowly to converge within 40 epochs. A batch size of 64 gives 125 updates per epoch.

For comparison, the same network is also trained with plain mini-batch SGD at the same learning rate.
""")

code(r"""
demo_cfg = make_config("demo_relu64_momentum", (64,), "relu", optimizer="momentum")
print("Momentum run:", demo_cfg)
demo = train(demo_cfg, X_train, y_train, X_val, y_val)
print(f"-> best val acc {demo['best_val_acc']:.4f} at epoch {demo['best_epoch']}, training time {demo['train_time_s']:.2f}s")

sgd_cfg = make_config("demo_relu64_sgd", (64,), "relu", optimizer="sgd")
print("\nPlain SGD run:", sgd_cfg)
demo_sgd = train(sgd_cfg, X_train, y_train, X_val, y_val)
print(f"-> best val acc {demo_sgd['best_val_acc']:.4f} at epoch {demo_sgd['best_epoch']}, training time {demo_sgd['train_time_s']:.2f}s")
""", tags=["report_hide_output"])

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for res, style, label in [(demo, "-", "momentum"), (demo_sgd, "--", "SGD")]:
    h = res["history"]
    axes[0].plot(h["epoch"], h["train_loss"], style, color="C0", label=f"train ({label})")
    axes[0].plot(h["epoch"], h["val_loss"], style, color="C1", label=f"val ({label})")
    axes[1].plot(h["epoch"], h["train_acc"], style, color="C0", label=f"train ({label})")
    axes[1].plot(h["epoch"], h["val_acc"], style, color="C1", label=f"val ({label})")
axes[0].set(title="Demo run: loss", xlabel="epoch", ylabel="cross-entropy")
axes[1].set(title="Demo run: accuracy", xlabel="epoch", ylabel="accuracy")
for ax in axes: ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md(r"""
## Hyperparameter tuning
* Hidden widths: **64, 128 and 256**, each tried with **sigmoid** and **ReLU**, giving 6 configurations.
* Fixed for every run: one hidden layer, momentum optimiser ($\beta=0.9$), $\eta=0.005$, batch size 64, 40 epochs, initialisation seed 0, shuffle seed 1.
* Selection uses the validation set only. The test set is not touched here.
""")

code(r"""
WIDTHS, ACTS = [64, 128, 256], ["sigmoid", "relu"]
results = []
for width in WIDTHS:
    for act in ACTS:
        cfg = make_config(f"{act}_h{width}", (width,), act)
        print(f"Run {len(results)}: {cfg['name']}")
        res = train(cfg, X_train, y_train, X_val, y_val)
        res["run_idx"] = len(results)
        results.append(res)
        print(f"-> best val acc {res['best_val_acc']:.4f} (epoch {res['best_epoch']}), {res['train_time_s']:.2f}s\n")
""", tags=["report_hide_output"])

md(r"""
## Question 3: Present and interpret the results

### (a) Results table
**How training time was measured:** wall-clock `time.perf_counter()` summed over the mini-batch forward, backward and update steps of all 40 epochs. Per-epoch evaluation and logging are excluded. All runs were on the same CPU with NumPy's default multithreaded BLAS, so absolute times vary between machines but the relative times are comparable.

The same table is logged to W&B as `results_table`.
""")

code(r"""
table = pd.DataFrame([{
    "run": r["run_idx"],
    "configuration": f"3072-{r['config']['hidden_sizes'][0]}-10 {r['config']['activation']}",
    "activation": r["config"]["activation"],
    "width": r["config"]["hidden_sizes"][0],
    "params": r["n_params"],
    "best_val_acc": r["best_val_acc"],
    "best_epoch": r["best_epoch"],
    "final_train_acc": r["history"]["train_acc"][-1],
    "final_val_acc": r["history"]["val_acc"][-1],
    "train_time_s": round(r["train_time_s"], 2),
} for r in results])
display(table.style.format({"params": "{:,}", "best_val_acc": "{:.4f}", "final_train_acc": "{:.4f}",
                            "final_val_acc": "{:.4f}", "train_time_s": "{:.2f}"})
        .hide(axis="index").hide(["activation", "width"], axis="columns"))

summary_run = wandb.init(project=WANDB_PROJECT, name="summary_table", mode=WANDB_MODE, reinit="finish_previous")
summary_run.log({"results_table": wandb.Table(dataframe=table)})
summary_run.finish()
""")

md(r"""
### (b) Model selection
The selection rule is the highest best-validation accuracy. Within a run, the earliest epoch that reached it is kept (enforced during training). Ties between runs go to the run with fewer parameters, and then to the earlier run.
""")

code(r"""
best_run = max(results, key=lambda r: (r["best_val_acc"], -r["n_params"], -r["run_idx"]))
print(f"Selected run {best_run['run_idx']}: {best_run['config']['name']} | "
      f"best val acc {best_run['best_val_acc']:.4f} at epoch {best_run['best_epoch']} | {best_run['n_params']:,} params")

# Persist the selected checkpoint to disk (restored in Question 4)
os.makedirs("checkpoints", exist_ok=True)
np.savez("checkpoints/best_model.npz", **best_run["best_params"])
with open("checkpoints/best_model_config.json", "w") as f:
    json.dump({**best_run["config"], "best_epoch": best_run["best_epoch"],
               "best_val_acc": best_run["best_val_acc"]}, f, indent=2)
""")

md("### (c) Learning curves of the selected run and comparison across runs")

code(r"""
h = best_run["history"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(h["epoch"], h["train_loss"], label="train"); axes[0].plot(h["epoch"], h["val_loss"], label="validation")
axes[1].plot(h["epoch"], h["train_acc"], label="train");  axes[1].plot(h["epoch"], h["val_acc"], label="validation")
for ax in axes:
    ax.axvline(best_run["best_epoch"], color="grey", ls=":", label=f"selected checkpoint (epoch {best_run['best_epoch']})")
    ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
axes[0].set(title=f"{best_run['config']['name']}: loss", ylabel="cross-entropy")
axes[1].set(title=f"{best_run['config']['name']}: accuracy", ylabel="accuracy")
plt.tight_layout(); plt.savefig("checkpoints/selected_run_curves.png", dpi=150); plt.show()
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(WIDTHS)); w = 0.38
for j, act in enumerate(ACTS):
    vals = [table.query("activation == @act and width == @wd")["best_val_acc"].item() for wd in WIDTHS]
    bars = ax.bar(x + (j - 0.5) * w, vals, w, label=act)
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
ax.set_xticks(x, [f"width {wd}" for wd in WIDTHS]); ax.set_ylabel("best validation accuracy")
ax.set_ylim(0.3, max(table.best_val_acc) + 0.03); ax.set_title("Best validation accuracy across runs")
ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("checkpoints/best_val_comparison.png", dpi=150); plt.show()
""")

code(r"""
# All runs' curves side by side (supports the discussion in (d))
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for r in results:
    hh, c = r["history"], r["config"]
    ls = "-" if c["activation"] == "relu" else "--"
    lbl = f"{c['activation']} {c['hidden_sizes'][0]}"
    color = f"C{WIDTHS.index(c['hidden_sizes'][0])}"
    axes[0].plot(hh["epoch"], hh["train_loss"], ls, color=color, label=lbl)
    axes[1].plot(hh["epoch"], hh["val_loss"], ls, color=color, label=lbl)
    axes[2].plot(hh["epoch"], hh["val_acc"], ls, color=color, label=lbl)
for ax, t in zip(axes, ["training loss", "validation loss", "validation accuracy"]):
    ax.set(title=t, xlabel="epoch"); ax.grid(alpha=.3)
axes[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("checkpoints/all_runs_curves.png", dpi=150); plt.show()
""")

md(r"""
### (d) Discussion: activation function, width, under/overfitting
| | width 64 | width 128 | width 256 |
|---|---|---|---|
| sigmoid: best val acc (epoch) | 0.4100 (39) | 0.4145 (39) | 0.4135 (40) |
| ReLU: best val acc (epoch) | 0.4290 (30) | 0.4345 (38) | **0.4420 (25)** |
| sigmoid: final train − val acc | 8.0 pp | 8.9 pp | 8.5 pp |
| ReLU: final train − val acc | 14.0 pp | 21.6 pp | 24.7 pp |

**Activation function.** ReLU beats sigmoid at every width, by about 2–3 percentage points, and it also learns faster: after the first epoch ReLU validation accuracy is 0.30–0.32 versus 0.25–0.27 for sigmoid. The sigmoid derivative is at most 0.25 and becomes very small when units saturate, so gradients reaching $W_1$ are damped. ReLU passes the gradient through unchanged for every active unit.

**Hidden-layer width.** Width barely matters for sigmoid: the three runs fall within 0.5 pp of each other, which is inside the epoch-to-epoch noise. For ReLU, accuracy rises steadily with width (0.429 → 0.435 → 0.442). The extra capacity costs about 4× the parameters (197k → 789k) and about 3.7× the training time (1.9 s → 7.0 s). Almost all of the parameters sit in the 3072 × width input layer.

**Underfitting or overfitting?**
* *Sigmoid runs mostly underfit or are under-trained.* Training and validation losses are still falling together at epoch 40, and the best checkpoints come from epochs 39–40. The train–validation gap stays small (about 8–9 pp). More epochs or a larger learning rate would probably still help.
* *ReLU runs overfit, and wider networks overfit more.* Training loss keeps falling (to 0.95 for width 256), while validation loss bottoms out around epochs 17–25 (≈1.61 for width 256) and then rises to ≈1.77. The train–validation accuracy gap widens from 14 pp at width 64 to 25 pp at width 256. This is why the selected checkpoint comes from epoch 25 and not from the end of training. Early stopping through checkpoint selection acts as the regulariser here. Weight decay, dropout or data augmentation would be the next steps.
* Even so, training accuracy remains modest for every model (49–67%). Much of the error is bias from the representation (see Q4c), not only variance.

**Noise caveat.** With momentum and batches of 64, validation accuracy fluctuates by about ±1–2 pp from one epoch to the next, and each configuration was run with one seed. Differences smaller than about 1 pp, such as between the sigmoid widths, should therefore not be over-interpreted.
""")

md(r"""
## Question 4: Evaluate the selected model
The checkpoint is restored from `checkpoints/best_model.npz` into a freshly constructed network and evaluated **once** on the 10,000 official test images. The model and its hyperparameters were fixed before this step and are not changed in response to the test results.
""")

code(r"""
with open("checkpoints/best_model_config.json") as f:
    sel_cfg = json.load(f)
final_model = MLP(sel_cfg["hidden_sizes"], sel_cfg["activation"])
with np.load("checkpoints/best_model.npz") as ckpt:
    final_model.params = {k: ckpt[k] for k in ckpt.files}

# sanity: restored model reproduces the recorded validation accuracy
_, restored_val_acc = evaluate(final_model, X_val, y_val)
assert np.isclose(restored_val_acc, sel_cfg["best_val_acc"])

test_loss, test_acc = evaluate(final_model, X_test, y_test)
y_pred = final_model.predict_proba(X_test).argmax(axis=1)
print(f"Selected model: {sel_cfg['name']} (epoch {sel_cfg['best_epoch']})")
print(f"(a) Test accuracy: {test_acc:.4f}   (test loss {test_loss:.4f})")

test_run = wandb.init(project=WANDB_PROJECT, name="final_test_eval", mode=WANDB_MODE, reinit="finish_previous",
                      config=sel_cfg)
test_run.summary.update({"test_acc": test_acc, "test_loss": test_loss})
test_run.finish()
""")

code(r"""
cm = confusion_matrix(y_test, y_pred, labels=range(10))
plt.figure(figsize=(9, 7.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar_kws={"label": "count"})
plt.xlabel("Predicted label"); plt.ylabel("True label")
plt.title(f"(b) Confusion matrix on CIFAR-10 test set — {sel_cfg['name']}, acc = {test_acc:.4f}")
plt.tight_layout(); plt.savefig("checkpoints/confusion_matrix.png", dpi=150); plt.show()

print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))
""")

md(r"""
### (c) Limitations of flattened pixels with a fully connected network
The selected model reaches **44.2% test accuracy**, which matches its validation accuracy, so model selection did not overfit the validation set. This is far above chance (10%) but well below simple CNNs (≥ 70–80%). The confusion matrix shows where the errors come from:

* **No spatial structure.** Flattening discards the 2-D neighbourhood of pixels. To the network, pixel (5, 5) and pixel (5, 6) are unrelated inputs, so it has to learn edges, textures and shapes from scratch without any built-in notion of locality.
* **No translation invariance or weight sharing.** Each first-layer weight is tied to one absolute pixel position. An object that shifts by a few pixels produces a completely different input vector, and the same pattern must be learned again at every location. This wastes parameters: 786k of the 789k weights sit in the first layer, and they still overfit 8,000 images, as shown by the train–validation gap in Q3.
* **Reliance on global colour and background.** Without local feature detectors the model falls back on coarse cues such as overall colour and layout. The typical confusions follow those cues: airplane → ship (184, blue sky versus blue sea), automobile ↔ truck (152 / 250), deer → bird / frog (181 / 115, green and brown backgrounds), and especially **dog → cat (421)**. Dog recall is only 14%, because the two classes differ in fine shape detail, not in colour statistics.
* **Sensitivity to nuisance factors.** Changes in lighting, scale, viewpoint or position move the raw pixel vector a long way, so classes are not close to separable in pixel space. The network has no prior to undo these changes, and with only 800 images per class and no augmentation it cannot learn that invariance from data either.
* **Poor scaling.** Larger images or deeper networks make the parameter count of fully connected layers explode. Convolutional layers, which use local receptive fields, shared weights and pooling, address all of the points above, but they are outside the scope of this assignment.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "assignment3.ipynb")
print("wrote assignment3.ipynb with", len(cells), "cells")
