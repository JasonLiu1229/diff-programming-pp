include("../libs/reverse.jl")

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
# TAPED SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

function simulate_lv(tape, slots)
    # Unpack input slots — the 6 parameters we differentiate w.r.t.
    s_α = slots[1]
    s_β = slots[2]
    s_γ = slots[3]
    s_δ = slots[4]
    s_x0 = slots[5]
    s_y0 = slots[6]

    # Constants — fixed slots, gradient will be zero for these
    s_dt = new_slot!(tape, DT)
    s_zero = new_slot!(tape, 0.0)

    # Running state slots
    s_x = s_x0
    s_y = s_y0
    s_loss = s_zero   # accumulate loss here

    for i = 1:STEPS
        # ── Lotka-Volterra Euler step ─────────────────────────────────────────
        #
        # dx = α*x - β*x*y
        s_αx = taped_mul(tape, s_α, s_x)          # α*x
        s_βxy = taped_mul(tape, s_β, s_x)          # β*x  (intermediate)
        s_βxy = taped_mul(tape, s_βxy, s_y)        # β*x*y
        s_dx = taped_sub(tape, s_αx, s_βxy)       # α*x - β*x*y

        # dy = -γ*y + δ*x*y
        s_γy = taped_mul(tape, s_γ, s_y)          # γ*y
        s_neg_γy = taped_sub(tape, s_zero, s_γy)    # -γ*y
        s_δxy = taped_mul(tape, s_δ, s_x)          # δ*x
        s_δxy = taped_mul(tape, s_δxy, s_y)        # δ*x*y
        s_dy = taped_add(tape, s_neg_γy, s_δxy)   # -γ*y + δ*x*y

        # x = x + dx * dt
        s_dx_dt = taped_mul(tape, s_dx, s_dt)
        s_x = taped_add(tape, s_x, s_dx_dt)

        # y = y + dy * dt
        s_dy_dt = taped_mul(tape, s_dy, s_dt)
        s_y = taped_add(tape, s_y, s_dy_dt)

        # ── Accumulate squared error into loss ────────────────────────────────
        #
        # Observation values at step i+1 are plain constants — new_slot! them
        s_ox = new_slot!(tape, OBS_X[i+1])
        s_oy = new_slot!(tape, OBS_Y[i+1])

        # ex = x - x_obs,  loss += ex²
        s_ex = taped_sub(tape, s_x, s_ox)
        s_ex2 = taped_mul(tape, s_ex, s_ex)
        s_loss = taped_add(tape, s_loss, s_ex2)

        # ey = y - y_obs,  loss += ey²
        s_ey = taped_sub(tape, s_y, s_oy)
        s_ey2 = taped_mul(tape, s_ey, s_ey)
        s_loss = taped_add(tape, s_loss, s_ey2)
    end

    return s_loss
end

# ─────────────────────────────────────────────────────────────────────────────
# REVERSE-MODE GRADIENT — one forward pass + one backward pass
# ─────────────────────────────────────────────────────────────────────────────

param_names = [
    "α (prey growth)",
    "β (predation)",
    "γ (pred. death)",
    "δ (pred. growth)",
    "x0 (init prey)",
    "y0 (init pred)",
]
params_perturbed = [1.1, 0.09, 1.4, 0.08, 10.5, 4.8]

println("=" ^ 60)
println("  Reverse-mode AD — Lotka-Volterra Parameter Fitting")
println("=" ^ 60)
println("  Model:  dx/dt = α·x - β·x·y")
println("          dy/dt = -γ·y + δ·x·y")
println()
println("  Parameters (perturbed from true values):")
true_vals = [1.0, 0.1, 1.5, 0.075, 10.0, 5.0]
for (name, val, tval) in zip(param_names, params_perturbed, true_vals)
    println("    $(rpad(name, 20)) = $val  (true: $tval)")
end
println()
println("  Running gradient() — ONE forward pass to build tape,")
println("  ONE backward pass to recover ALL $(length(params_perturbed)) gradients...")
println()

grads = gradient(simulate_lv, params_perturbed)

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
println("  Gradients (reverse-mode vs finite differences):")
println(
    "  $(rpad("Parameter", 22)) $(rpad("Reverse AD", 14)) $(rpad("Finite Diff", 14)) Match?",
)
println("  " * "-"^60)
for i = 1:length(params_perturbed)
    rev = round(grads[i], digits = 4)
    fd = round(fd_grads[i], digits = 4)
    ok = isapprox(grads[i], fd_grads[i], rtol = 1e-3) ? "✓" : "✗"
    println(
        "  $(rpad(param_names[i], 22)) $(rpad(string(rev), 14)) $(rpad(string(fd), 14)) $ok",
    )
end
println()
println("  Forward passes executed : 1  (tape built once)")
println("  Backward passes executed: 1  (all gradients at once)")
println("  Inputs (n)              : $(length(params_perturbed))")
println("=" ^ 60)
