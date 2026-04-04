include("../libs/forward.jl")

const DT = 0.01    # Euler time step
const STEPS = 400     # simulate 4.0 time units

function generate_observations(α, β, γ, δ, x0, y0)
    xs = Float64[x0]
    ys = Float64[y0]
    x, y = x0, y0
    for _ = 1:STEPS
        dx = α*x - β*x*y
        dy = -γ*y + δ*x*y
        x = x + dx * DT
        y = y + dy * DT
        push!(xs, x)
        push!(ys, y)
    end
    return xs, ys
end

# True parameters — what we are trying to recover
const TRUE_α = 1.0
const TRUE_β = 0.1
const TRUE_γ = 1.5
const TRUE_δ = 0.075
const TRUE_x0 = 10.0
const TRUE_y0 = 5.0

const OBS_X, OBS_Y = generate_observations(TRUE_α, TRUE_β, TRUE_γ, TRUE_δ, TRUE_x0, TRUE_y0)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION  
# ─────────────────────────────────────────────────────────────────────────────

function simulate_lv(α, β, γ, δ, x0, y0)
    x = x0
    y = y0
    loss = x * 0.0   # zero with correct type — ensures loss is Dual when inputs are

    for i = 1:STEPS
        # Lotka-Volterra Euler step
        dx = α*x - β*x*y
        dy = -γ*y + δ*x*y
        x = x + dx * DT
        y = y + dy * DT

        # Accumulate squared error against observations at each step
        ex = x - OBS_X[i+1]
        ey = y - OBS_Y[i+1]
        loss = loss + ex*ex + ey*ey
    end

    return loss
end

# ─────────────────────────────────────────────────────────────────────────────
# FORWARD-MODE GRADIENT
# ─────────────────────────────────────────────────────────────────────────────

param_names = [
    "α (prey growth)",
    "β (predation)",
    "γ (pred. death)",
    "δ (pred. growth)",
    "x0 (init prey)",
    "y0 (init pred)",
]
params = [1.0, 0.1, 1.5, 0.075, 10.0, 5.0]   # start at true values → loss ≈ 0, grads ≈ 0

# Perturb slightly so gradients are non-trivial and differences are visible
params_perturbed = [1.1, 0.09, 1.4, 0.08, 10.5, 4.8]

println("=" ^ 60)
println("  Forward-mode AD — Lotka-Volterra Parameter Fitting")
println("=" ^ 60)
println("  Model:  dx/dt = α·x - β·x·y")
println("          dy/dt = -γ·y + δ·x·y")
println()
println("  Parameters (perturbed from true values):")
for (name, val, tval) in zip(param_names, params_perturbed, params)
    println("    $(rpad(name, 20)) = $val  (true: $tval)")
end
println()
println("  Running dual_gradient — calls simulate_lv once per input...")
println(
    "  That is $(length(params_perturbed)) forward passes for $(length(params_perturbed)) parameters.",
)
println()

passes_run = 0
grads = dual_gradient(params_perturbed) do args...
    global passes_run += 1
    simulate_lv(args...)
end

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION via central finite differences
# ─────────────────────────────────────────────────────────────────────────────

function simulate_lv_plain(α, β, γ, δ, x0, y0)
    x, y, loss = x0, y0, 0.0
    for i = 1:STEPS
        dx = α*x - β*x*y
        dy = -γ*y + δ*x*y
        x = x + dx * DT
        y = y + dy * DT
        ex = x - OBS_X[i+1]
        ey = y - OBS_Y[i+1]
        loss += ex^2 + ey^2
    end
    return loss
end

ε = 1e-5
fd_grads = Float64[]
for i = 1:length(params_perturbed)
    p_plus = copy(params_perturbed);
    p_plus[i] += ε
    p_minus = copy(params_perturbed);
    p_minus[i] -= ε
    fd = (simulate_lv_plain(p_plus...) - simulate_lv_plain(p_minus...)) / (2ε)
    push!(fd_grads, fd)
end

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

loss_val = simulate_lv_plain(params_perturbed...)
println("  Loss L = Σ(x_pred - x_obs)² + (y_pred - y_obs)²")
println("         = $(round(loss_val, digits=4))")
println()
println("  Gradients (forward-mode vs finite differences):")
println(
    "  $(rpad("Parameter", 22)) $(rpad("Forward AD", 14)) $(rpad("Finite Diff", 14)) Match?",
)
println("  " * "-"^60)
for i = 1:length(params_perturbed)
    fwd = round(grads[i], digits = 4)
    fd = round(fd_grads[i], digits = 4)
    ok = isapprox(grads[i], fd_grads[i], rtol = 1e-3) ? "✓" : "✗"
    println(
        "  $(rpad(param_names[i], 22)) $(rpad(string(fwd), 14)) $(rpad(string(fd), 14)) $ok",
    )
end
println()
println("  Forward passes executed : $passes_run  (one per input)")
println("  Inputs (n)              : $(length(params_perturbed))")
println("=" ^ 60)
