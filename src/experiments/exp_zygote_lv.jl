using Zygote

# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM SETUP  
# ─────────────────────────────────────────────────────────────────────────────

const DT = 0.01
const STEPS = 400

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
    loss = zero(α)      # zero() picks the right type — works for Float64 and Zygote's
    # internal dual-like types alike. Safer than x0 * 0.0 here.
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

# ─────────────────────────────────────────────────────────────────────────────
# ZYGOTE GRADIENT
# ─────────────────────────────────────────────────────────────────────────────
#
# Zygote.gradient takes a function and its arguments, and returns a tuple of
# gradients — one per argument. This is reverse mode: one pass to build the
# internal computation graph, one backward pass to get all 6 gradients.
#
# The API looks different from our manual reverse (gradient(f, [params...])),
# because Zygote works with individual scalar arguments, not a vector of slots.
# We wrap simulate_lv in a closure that unpacks a vector so we can loop over
# parameter counts easily in the benchmarking file.

param_names = [
    "α (prey growth)",
    "β (predation)",
    "γ (pred. death)",
    "δ (pred. growth)",
    "x0 (init prey)",
    "y0 (init pred)",
]
params_perturbed = [1.1, 0.09, 1.4, 0.08, 10.5, 4.8]
true_vals = [1.0, 0.1, 1.5, 0.075, 10.0, 5.0]

println("=" ^ 60)
println("  Zygote — Lotka-Volterra Parameter Fitting")
println("=" ^ 60)
println("  Model:  dx/dt = α·x - β·x·y")
println("          dy/dt = -γ·y + δ·x·y")
println()
println("  Backend: Zygote.jl  (source transformation, Julia IR level)")
println()
println("  Parameters (perturbed from true values):")
for (name, val, tval) in zip(param_names, params_perturbed, true_vals)
    println("    $(rpad(name, 20)) = $val  (true: $tval)")
end
println()

# Zygote.gradient returns a tuple of gradients matching each argument
grads_tuple = Zygote.gradient(simulate_lv, params_perturbed...)

# Collect into a plain vector so our output loop works the same as other files
grads = collect(grads_tuple)

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
println("  Gradients (Zygote vs finite differences):")
println(
    "  $(rpad("Parameter", 22)) $(rpad("Zygote", 14)) $(rpad("Finite Diff", 14)) Match?",
)
println("  " * "-"^60)
for i = 1:length(params_perturbed)
    zy = round(grads[i], digits = 4)
    fd = round(fd_grads[i], digits = 4)
    ok = isapprox(grads[i], fd_grads[i], rtol = 1e-3) ? "✓" : "✗"
    println(
        "  $(rpad(param_names[i], 22)) $(rpad(string(zy), 14)) $(rpad(string(fd), 14)) $ok",
    )
end
println("=" ^ 60)
