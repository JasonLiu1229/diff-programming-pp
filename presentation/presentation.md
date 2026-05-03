---
theme:
  name: dark
  override:
    default:
      colors:
        foreground: "e2e8f0"
        background: "0f172a"
    slide_title:
      colors:
        foreground: "7dd3fc"
      alignment: left
    footer:
      style: template
      left: "Jason Liu — University of Antwerp"
      right: "{current_slide} / {total_slides}"
---

<!-- jump_to_middle -->

# Choosing the Right Derivative

### A Practical Comparison of Julia's AD Strategies

---

**Jason Liu** — Master Student Software Engineering
University of Antwerp · Programming Paradigms

<!-- end_slide -->

# What Is Differentiable Programming?

<!-- pause -->

> A paradigm where programs are written so that **derivatives can be computed through them automatically**

This enables **gradient-based optimization** — the backbone of:

- Machine learning (training neural networks)
- Scientific parameter estimation
- Numerical simulation

The key tool that makes this possible is **Automatic Differentiation (AD)**

<!-- pause -->

### Why not just use calculus by hand?

<!-- pause -->

Real programs have thousands of parameters. Hand-deriving gradients is infeasible.

<!-- end_slide -->

# Automatic Differentiation — Two Modes

<!-- pause -->

### Forward Mode
- Attaches a **derivative component** to every value during computation
- Requires **n passes** for a function with n inputs
- Cheap per pass, but **scales linearly** with input count

<!-- pause -->

### Reverse Mode
- Records every operation onto a **"tape"** during a forward pass
- Recovers **all n gradients** in a single backward sweep
- One backward pass = all gradients simultaneously
- Preferred when n is large and output is a scalar

<!-- pause -->

```
Forward:  O(n) passes  →  good for few inputs
Reverse:  O(1) passes  →  good for many inputs
```

<!-- end_slide -->

# The Experiment: Lotka-Volterra ODE

<!-- pause -->

We use a **predator-prey model** as our test vehicle:
 
```
dx/dt = α·x − β·x·y    (prey)
dy/dt = δ·x·y − γ·y    (predators)
```

<!-- pause -->

**Problem framing:** Parameter fitting
- Given observed population trajectories
- Find parameters minimizing a **least-squares loss**
- Compute the **gradient of that loss** using AD


<!-- pause -->

```julia
function simulate_lv_scalar(α, β, γ, δ, x0, y0)
    x = x0;
    y = y0;
    loss = zero(α)
    for i = 1:STEPS
        dx = α*x - β*x*y
        dy = -γ*y + δ*x*y
        x = x + dx * DT
        y = y + dy * DT
        ex = x - OBS_X[i+1]
        ey = y - OBS_Y[i+1]
        loss = loss + ex*ex + ey*ey
    end
    return loss
end
```

<!-- pause -->


We compare **4 AD strategies** in Julia:
`Forward (Dual)` · `Reverse (Tape)` · `Zygote.jl` · `Enzyme.jl`

<!-- end_slide -->

# Strategy 1 — Forward Mode with Dual Numbers

A **dual number** carries two components: the value and its derivative

```julia
struct Dual
    value::Float64
    deriv::Float64
end
```

<!-- pause -->

Overload all arithmetic operators → chain rule propagates automatically

```julia
# Differentiating w.r.t. parameter i:
# seed input i with deriv=1.0, all others with 0.0
dual_gradient(p) do args...
    simulate_lv_scalar(args...)
end
```

**Key insight:** The simulation code is **unchanged** — the Dual type handles everything transparently, just like operator overloading in OOP.

<!-- end_slide -->

# Strategy 2 — Reverse Mode with a Tape

**Phase 1 — Forward pass:**
Run the computation, but **record every operation** onto a tape. Save input values — you'll need them in the backward pass.

<!-- pause -->

**Phase 2 — Backward pass:**
Seed output gradient = 1.0, walk the tape **backwards**, apply local derivative rules, accumulate gradients at each input.

```julia
s_αx  = taped_mul(tape, s_α, s_x)
s_βxy = taped_mul(tape, taped_mul(tape, s_β, s_x), s_y)
s_dx  = taped_sub(tape, s_αx, s_βxy)
```

<!-- pause -->

⚠️ Every natural expression must be manually routed through the tape. Two readable lines become ten lines of tape manipulation.

<!-- end_slide -->

# Strategy 3 — Zygote.jl

**Source-to-source transformation:** Zygote parses your Julia code and **generates new Julia code** that computes gradients.

```
Julia function  →  Zygote (source-to-source)  →  New gradient function
```

<!-- pause -->

Usage is one line — the simulation is completely unchanged:

```julia
Zygote.gradient(simulate_lv_scalar, p...)
```

<!-- pause -->

**The catch:** Zygote generates the backward pass as a **chain of closures** — each is a heap allocation. For 400 timesteps × ~10 ops/step = thousands of heap objects per gradient call. The garbage collector has to clean these up.


<!-- pause -->

```julia
function add(a, b)
    return a + b
```

<!-- pause -->

```julia
x = 10

add_x = (a) -> a + x   # x is not a parameter, but it's remembered
add_x(5)               # returns 15
add_x(20)              # returns 30
```

<!-- end_slide -->

# Strategy 4 — Enzyme.jl

Enzyme operates at the **LLVM IR level** — below Julia's object system entirely.

```
Julia function  →  LLVM IR  →  Enzyme differentiates IR  →  Gradients
```

<!-- pause -->

- Intermediate values live in **CPU registers**, not the heap
- **No memory allocation** during the backward pass
- You can choose forward or reverse mode explicitly

```julia
d = zeros(Float64, length(p))
Enzyme.autodiff(ReverseWithPrimal,
    simulate_lv_vector, Active,
    Duplicated(p, d))
```

<!-- pause -->

Requires slight API adaptation (vector args + shadow array annotation), but remains reasonably concise.

<!-- end_slide -->

# Results — Correctness

All four methods were validated against **central finite differences**:

```
∂L/∂θᵢ ≈ [L(θᵢ + ε) − L(θᵢ − ε)] / 2ε     (ε = 1e-5)
```

| Method | Max Absolute Error | Max Relative Error |
|---|---|---|
| Forward (Dual) | 0.161 | 7.24e-8 |
| Reverse (Tape) | 0.161 | 7.24e-8 |
| Zygote | 0.161 | 7.24e-8 |
| Enzyme | 0.161 | 7.24e-8 |

✅ All four agree to **machine precision** — timing comparisons are between equally correct implementations.

<!-- end_slide -->

# Results — Timing (6 Parameters)

| Method | Median | Min | Std |
|---|---|---|---|
| **Enzyme** | **5.5 μs** | 4.2 μs | 7.36 μs |
| Forward (Dual) | 15.3 μs | 13.6 μs | 15.9 μs |
| Reverse (Tape) | 1.767 ms | 393.4 μs | 3.797 ms |
| Zygote | 5.364 ms | 1.922 ms | 6.685 ms |

<!-- pause -->

- **Enzyme** is ~3× faster than forward mode, ~1000× faster than Zygote
- **Forward mode** is second despite being simplest — 6 lightweight passes
- **Tape** is slow: every op allocates a `TapeEntry` struct on the heap
- **Zygote** is slowest: thousands of closure allocations + GC pressure (high std dev!)

<!-- end_slide -->

# Results — Scaling Behaviour

As the number of parameters grows (n = 4 → 24):

| Params | Forward | Reverse | Zygote | Enzyme |
|---|---|---|---|---|
| 4 | 1.07 ms | 2.14 ms | 79 ms | 0.10 ms |
| 8 | 5.36 ms | 3.80 ms | 140 ms | 0.20 ms |
| 16 | 18.12 ms | 7.15 ms | 265 ms | 0.27 ms |
| 24 | 36.01 ms | 11.04 ms | 406 ms | 0.40 ms |

<!-- pause -->

- **Enzyme** stays nearly flat — values in registers, no allocation growth
- **Reverse (Tape)** is broadly O(1) in parameters — one backward pass regardless; crossover with forward mode at ~n=8
- **Forward** grows linearly — O(n) passes, one per parameter
- **Zygote** also linear but with a much higher baseline due to heap cost

<!-- end_slide -->

# Summary — When to Use What?

| | Enzyme | Forward | Reverse (Tape) | Zygote |
|---|---|---|---|---|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Scalability** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Simplicity** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |



<!-- end_slide -->

# References & Code

Scan to access the full bibliography:

```
█████████████████████████████████████████
██ ▄▄▄▄▄ █▀ █▀▀▀▄▀▄█▄▄▄▄▀▄▀▄▀▀ █ ▄▄▄▄▄ ██
██ █   █ █▀ ▄ █▀▄▀▄▀█▄▀▄█▄▀▀▄█▄█ █   █ ██
██ █▄▄▄█ █▀█ █▄ ▀▀ ▀▄▄ ▀▀▄█ ▀▀ █ █▄▄▄█ ██
██▄▄▄▄▄▄▄█▄█▄█ █▄▀ █ ▀ ▀ █ █ █▄█▄▄▄▄▄▄▄██
██   ▄ ▀▄▄▄ ▄█▄▄▄▄▀▄█ █▀  ▄▀▄▄▀ █ █ ▀ ███
███▀█ ▄█▄█ ▄▄ ▄▄█▄█▄▄██ ██▄▄█▄▀  █▄ ▄▀▄██
██▀▄█ ▄ ▄ █▄ ▄▀ ▄▄   ▀ ▀ ▀▄▄▄██ ▄▀▄▄ ▀▄██
███▀  ▀█▄▀  ▀█▀█▀▄ ▄███▀█▀██▀ ▀▀▄ █▀▄▄ ██
██ ▀▀▄█▄▄▄▀▄ █▄▄▄██ ▀▀ █▄  ▄ ▄ ▄▄ ▄▄ ▀▄██
██▀▄▀▄█ ▄█▀ █ ▄▄█▄▀█ █ ▀ █▀▄▀▄▀  █▀█▄▄ ██
██▀█▄ ▀ ▄▀▄▄█▄▀▀██▀▄▀ █▀▄ ▀ ▄█▀ ▄▀▄▀▄ ▄██
██  ▄█▀▀▄▀▀█▄█▀ ▄ ▄██▀ ▀ ▀ ▄   █▄▀█▄▄▄ ██
██▀ ▀ █▄▄ ▀  █▄█▄█▀▄▀▀▀▀ ▀▀ ▄▄▀▄▄ ▄ ▄ ▄██
██ █ ▄ ▄▄█▀▀█ ▄█▀ ▄█▄▀▀▀█▀ ▄▄ ▀▀▄▄▀██▄ ██
██▄█████▄█▀▄ ▄▀▀▀▄▀▄▀ ▀▀▄▀▀▀▄▄ ▄▄▄ █▄████
██ ▄▄▄▄▄ █▄ ██▀ ▄  █▀█▄ ▀█▀█▄█ █▄█ ▀▄ ▄██
██ █   █ █  ██▄█▄█▀▄█ ▀█▄▀▀ ▄    ▄ ▄ ▀▀██
██ █▄▄▄█ █  ▄ ▄█▀▄ ▄▄█   ███▀▀ ▀█ ▀▄▄▄ ██
██▄▄▄▄▄▄▄█▄██▄█████▄▄▄██▄██▄▄██▄█▄█▄▄▄▄██
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
```

`https://github.com/JasonLiu1229/diff-programming-pp`

_Jason Liu · jason.liu@student.uantwerpen.be_
_University of Antwerp — Programming Paradigms_
