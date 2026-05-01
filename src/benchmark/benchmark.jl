using BenchmarkTools
using Plots
using Statistics
using Zygote
using Enzyme

include("../libs/forward.jl")
include("../libs/reverse.jl")

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK DESIGN OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
#
# We measure four things for each AD method:
#
#   1. TIMING       — how long does one gradient computation take?
#                     We use BenchmarkTools.@benchmark which runs the function
#                     many times, discards the first run (JIT warmup), and
#                     reports min/median/mean/std. A single @elapsed is
#                     unreliable in Julia because the first call always compiles.
#
#   2. SCALING      — how does timing grow as we add more parameters?
#                     We extend Lotka-Volterra to n species (generalised LV)
#                     and sweep n from 2 to 20. Forward mode should grow
#                     linearly O(n); reverse mode should stay roughly O(1).
#
#   3. ACCURACY     — do all methods agree with finite differences?
#                     We report the max absolute error across all 6 gradients.

# ─────────────────────────────────────────────────────────────────────────────
# SHARED SETUP — observations and parameters used by every method
# ─────────────────────────────────────────────────────────────────────────────

const DT = 0.01
const STEPS = 400

const TRUE_α = 1.0
const TRUE_β = 0.1
const TRUE_γ = 1.5
const TRUE_δ = 0.075
const TRUE_x0 = 10.0
const TRUE_y0 = 5.0

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

const OBS_X, OBS_Y = generate_observations(TRUE_α, TRUE_β, TRUE_γ, TRUE_δ, TRUE_x0, TRUE_y0)

# Perturbed parameters — what we differentiate at
const PARAMS = [1.1, 0.09, 1.4, 0.08, 10.5, 4.8]
const PARAM_NAMES = ["α", "β", "γ", "δ", "x0", "y0"]

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION FUNCTIONS — one version per API style
# ─────────────────────────────────────────────────────────────────────────────

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

function simulate_lv_vector(params::Vector{Float64})
    α, β, γ, δ, x0, y0 = params[1], params[2], params[3], params[4], params[5], params[6]
    x = x0;
    y = y0;
    loss = 0.0
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

# Tape-based version for manual reverse mode
function simulate_lv_tape(tape, slots)
    s_α=slots[1];
    s_β=slots[2];
    s_γ=slots[3]
    s_δ=slots[4];
    s_x0=slots[5];
    s_y0=slots[6]
    s_dt=new_slot!(tape, DT);
    s_zero=new_slot!(tape, 0.0)
    s_x=s_x0;
    s_y=s_y0;
    s_loss=s_zero
    for i = 1:STEPS
        s_αx = taped_mul(tape, s_α, s_x)
        s_βxy = taped_mul(tape, taped_mul(tape, s_β, s_x), s_y)
        s_dx = taped_sub(tape, s_αx, s_βxy)
        s_γy = taped_mul(tape, s_γ, s_y)
        s_negγy = taped_sub(tape, s_zero, s_γy)
        s_δxy = taped_mul(tape, taped_mul(tape, s_δ, s_x), s_y)
        s_dy = taped_add(tape, s_negγy, s_δxy)
        s_x = taped_add(tape, s_x, taped_mul(tape, s_dx, s_dt))
        s_y = taped_add(tape, s_y, taped_mul(tape, s_dy, s_dt))
        s_ox=new_slot!(tape, OBS_X[i+1]);
        s_oy=new_slot!(tape, OBS_Y[i+1])
        s_ex=taped_sub(tape, s_x, s_ox);
        s_ey=taped_sub(tape, s_y, s_oy)
        s_loss=taped_add(
            tape,
            s_loss,
            taped_add(tape, taped_mul(tape, s_ex, s_ex), taped_mul(tape, s_ey, s_ey)),
        )
    end
    return s_loss
end

# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT WRAPPERS — one callable per method, same interface
# ─────────────────────────────────────────────────────────────────────────────
function grad_forward(p::Vector{Float64})
    dual_gradient(p) do args...
        simulate_lv_scalar(args...)
    end
end

function grad_reverse(p::Vector{Float64})
    gradient(simulate_lv_tape, p)
end

function grad_zygote(p::Vector{Float64})
    collect(Zygote.gradient(simulate_lv_scalar, p...))
end

function grad_enzyme(p::Vector{Float64})
    d = zeros(Float64, length(p))
    Enzyme.autodiff(ReverseWithPrimal, simulate_lv_vector, Active, Duplicated(p, d))
    return copy(d)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CORRECTNESS CHECK
# ─────────────────────────────────────────────────────────────────────────────
println("\n", "=" ^ 65)
println("  SECTION 1 — Correctness")
println("=" ^ 65)

ε = 1e-5
fd_grads = map(1:length(PARAMS)) do i
    p_plus = copy(PARAMS);
    p_plus[i] += ε
    p_minus = copy(PARAMS);
    p_minus[i] -= ε
    (simulate_lv_vector(p_plus) - simulate_lv_vector(p_minus)) / (2ε)
end

# Warm up JIT for all methods before measuring anything
_ = grad_forward(PARAMS)
_ = grad_reverse(PARAMS)
_ = grad_zygote(PARAMS)
_ = grad_enzyme(PARAMS)

methods = ["Forward (Dual)", "Reverse (Tape)", "Zygote", "Enzyme"]
grad_fns = [grad_forward, grad_reverse, grad_zygote, grad_enzyme]

println("  Max |AD_grad - FD_grad| across all 6 parameters:\n")
println("  $(rpad("Method", 20))  $(rpad("Max Abs Err", 14))  Max Rel Err")
println("  " * "-"^55)

all_grads = Dict{String,Vector{Float64}}()
for (name, fn) in zip(methods, grad_fns)
    g = fn(PARAMS)
    all_grads[name] = g
    abs_err = maximum(abs.(g .- fd_grads))
    # Relative error: divide by magnitude of FD gradient (avoid div by zero)
    rel_err = maximum(abs.(g .- fd_grads) ./ (abs.(fd_grads) .+ 1e-10))
    println(
        "  $(rpad(name, 20))  $(rpad(round(abs_err, sigdigits=3), 14))  $(round(rel_err, sigdigits=3))",
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TIMING BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
#
# @benchmark runs the expression in a loop, automatically deciding how many
# samples to take (minimum 1 second of total runtime by default). It returns
# a Trial object containing all sample times. We extract:
#
#   minimum  — best case, least OS noise, most representative of raw speed
#   median   — robust central estimate, less affected by outliers than mean
#   std      — spread — a high std means the timing is noisy/unreliable
#
# We use $PARAMS with the $ interpolation to avoid benchmarking the lookup
# of the global variable on every call — a common BenchmarkTools gotcha.

println("\n", "=" ^ 65)
println("  SECTION 2 — Timing (BenchmarkTools, median of many runs)")
println("=" ^ 65)
println()
println("  $(rpad("Method", 20))  $(rpad("Median", 12))  $(rpad("Min", 12))  Std")
println("  " * "-"^60)

timing_results = Dict{String,BenchmarkTools.Trial}()
for (name, fn) in zip(methods, grad_fns)
    trial = @benchmark $fn($PARAMS) samples=200 evals=1
    timing_results[name] = trial
    med = BenchmarkTools.prettytime(median(trial).time)
    mn = BenchmarkTools.prettytime(minimum(trial).time)
    sd = BenchmarkTools.prettytime(std(trial.times))
    println("  $(rpad(name, 20))  $(rpad(med, 12))  $(rpad(mn, 12))  $sd")
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SCALING WITH NUMBER OF PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
#
# This is the core experiment. We extend Lotka-Volterra to a generalised
# n-species system where each species interacts with its two neighbours.
# This gives us a clean way to increase the number of parameters (n growth
# rates + n interaction terms = 2n parameters total) while keeping the
# simulation structure the same.
#
# We expect:
#   Forward mode  →  O(n) time, because it runs one full pass per parameter
#   Reverse mode  →  O(1) time w.r.t. n, because one backward pass gets all grads
#   Zygote        →  O(1) similarly
#   Enzyme        →  O(1) similarly, and likely fastest due to LLVM optimisation

println("\n", "=" ^ 65)
println("  SECTION 3 — Scaling with number of parameters")
println("=" ^ 65)
println()
println("  We sweep species count n from 2 to 12.")
println("  Each system has 2n parameters (n growth rates + n interaction terms).")
println("  Timing is median over 50 samples at each n.\n")

# Generalised Lotka-Volterra: n prey species, each grows at rate r[i]
# and is suppressed by interaction with the next species at rate a[i].
# This is a clean n-parameter system with the same ODE structure.

function make_glv_forward(n_species)
    # Returns a gradient function for a system with 2*n_species parameters
    # params = [r1,...,rn, a1,...,an]
    function f(params_vec)
        n = n_species
        # Build n independent OBS trajectories (just use the same sine wave
        # pattern scaled by species index — we need *something* to fit against)
        function grad(p)
            dual_gradient(p) do args...
                rs = args[1:n]
                as = args[(n+1):2n]
                # T is Dual when forward mode runs, Float64 otherwise.
                # similar(xs) would lock the type to Float64 at construction
                # time and crash when forward mode tries to store a Dual into it.
                T = typeof(args[1])
                xs = T[10.0 + i for i = 1:n]
                loss = zero(args[1])
                for step = 1:STEPS
                    new_xs = Vector{T}(undef, n)
                    for i = 1:n
                        j = mod1(i+1, n)
                        dxi = rs[i]*xs[i] - as[i]*xs[i]*xs[j]
                        new_xs[i] = xs[i] + dxi * DT
                    end
                    xs = new_xs
                    for i = 1:n
                        target = 10.0 + i + sin(step * DT)  # synthetic target
                        e = xs[i] - target
                        loss = loss + e*e
                    end
                end
                return loss
            end
        end
        return grad
    end
    return f(nothing)
end

function make_glv_reverse(n_species)
    function grad(p::Vector{Float64})
        n = n_species
        function tape_fn(tape, slots)
            rs = slots[1:n]
            as = slots[(n+1):2n]
            s_dt = new_slot!(tape, DT)
            s_zero = new_slot!(tape, 0.0)
            s_xs = [new_slot!(tape, 10.0 + Float64(i)) for i = 1:n]
            s_loss = s_zero
            for step = 1:STEPS
                new_s_xs = Vector{Int}(undef, n)
                for i = 1:n
                    j = mod1(i+1, n)
                    s_rixi = taped_mul(tape, rs[i], s_xs[i])
                    s_xixj = taped_mul(tape, s_xs[i], s_xs[j])
                    s_aixij = taped_mul(tape, as[i], s_xixj)
                    s_dxi = taped_sub(tape, s_rixi, s_aixij)
                    new_s_xs[i] = taped_add(tape, s_xs[i], taped_mul(tape, s_dxi, s_dt))
                end
                s_xs = new_s_xs
                for i = 1:n
                    target_val = 10.0 + i + sin(step * DT)
                    s_tgt = new_slot!(tape, target_val)
                    s_e = taped_sub(tape, s_xs[i], s_tgt)
                    s_loss = taped_add(tape, s_loss, taped_mul(tape, s_e, s_e))
                end
            end
            return s_loss
        end
        return gradient(tape_fn, p)
    end
    return grad
end

function make_glv_zygote(n_species)
    function grad(p::Vector{Float64})
        n = n_species
        function f(args...)
            rs = args[1:n];
            as = args[(n+1):2n]
            xs = [10.0 + Float64(i) for i = 1:n]
            loss = zero(args[1])
            for step = 1:STEPS
                new_xs = map(1:n) do i
                    j = mod1(i+1, n)
                    xs[i] + (rs[i]*xs[i] - as[i]*xs[i]*xs[j]) * DT
                end
                xs = new_xs
                for i = 1:n
                    e = xs[i] - (10.0 + i + sin(step * DT))
                    loss = loss + e*e
                end
            end
            return loss
        end
        collect(Zygote.gradient(f, p...))
    end
    return grad
end

function make_glv_enzyme(n_species)
    function grad(p::Vector{Float64})
        n = n_species
        function f(pv::Vector{Float64})
            rs = pv[1:n];
            as = pv[(n+1):2n]
            xs = [10.0 + Float64(i) for i = 1:n]
            loss = 0.0
            for step = 1:STEPS
                new_xs = similar(xs)
                for i = 1:n
                    j = mod1(i+1, n)
                    new_xs[i] = xs[i] + (rs[i]*xs[i] - as[i]*xs[i]*xs[j]) * DT
                end
                xs = new_xs
                for i = 1:n
                    e = xs[i] - (10.0 + i + sin(step * DT))
                    loss += e*e
                end
            end
            return loss
        end
        d = zeros(Float64, length(p))
        Enzyme.autodiff(ReverseWithPrimal, f, Active, Duplicated(p, d))
        return copy(d)
    end
    return grad
end

# Sweep n_species from 2 to 12
species_counts = 2:2:12
forward_times = Float64[]
reverse_times = Float64[]
zygote_times = Float64[]
enzyme_times = Float64[]

for n in species_counts
    p = vcat(fill(1.0, n), fill(0.1, n))   # 2n parameters: n rates + n interactions

    gf = make_glv_forward(n)
    gr = make_glv_reverse(n)
    gz = make_glv_zygote(n)
    ge = make_glv_enzyme(n)

    # Warmup each method to trigger JIT compilation before timing
    gf(p);
    gr(p);
    gz(p);
    ge(p)

    push!(forward_times, median(@benchmark $gf($p) samples=50 evals=1).time / 1e6)  # ns → ms
    push!(reverse_times, median(@benchmark $gr($p) samples=50 evals=1).time / 1e6)
    push!(zygote_times, median(@benchmark $gz($p) samples=50 evals=1).time / 1e6)
    push!(enzyme_times, median(@benchmark $ge($p) samples=50 evals=1).time / 1e6)

    println(
        "  n=$(lpad(2n, 3)) params | fwd=$(rpad(round(forward_times[end],digits=2),7))ms  " *
        "rev=$(rpad(round(reverse_times[end],digits=2),7))ms  " *
        "zyg=$(rpad(round(zygote_times[end],digits=2),7))ms  " *
        "enz=$(round(enzyme_times[end],digits=2))ms",
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────
#
# We produce three plots:
#
#   Plot 1 — LV Trajectory
#     Shows true vs perturbed population trajectories over time.
#     This is the visual motivation for the experiment — the gap between
#     curves is literally what the loss function measures.
#
#   Plot 2 — Scaling (time vs number of parameters)
#     The main scientific result. Forward mode grows linearly; the others
#     stay roughly flat. This is the core argument of the paper.
#
#   Plot 3 — Timing bar chart for the standard 6-parameter case
#     Clean side-by-side comparison of all four methods at the same task.

println("\n", "=" ^ 65)
println("  SECTION 4 — Generating plots...")
println("=" ^ 65)

# ── Plot 1: LV Trajectory ────────────────────────────────────────────────────

function run_trajectory(α, β, γ, δ, x0, y0)
    xs = Float64[x0];
    ys = Float64[y0]
    x, y = x0, y0
    for _ = 1:STEPS
        dx = α*x - β*x*y
        dy = -γ*y + δ*x*y
        x = x + dx * DT
        y = y + dy * DT
        push!(xs, x);
        push!(ys, y)
    end
    return xs, ys
end

t_range = range(0.0, step = DT, length = STEPS+1)

true_x, true_y = run_trajectory(TRUE_α, TRUE_β, TRUE_γ, TRUE_δ, TRUE_x0, TRUE_y0)
pert_x, pert_y = run_trajectory(PARAMS...)

p1 = plot(
    t_range,
    true_x,
    label = "Prey (true params)",
    color = :steelblue,
    lw = 2,
    linestyle = :solid,
)
plot!(
    p1,
    t_range,
    true_y,
    label = "Predator (true params)",
    color = :firebrick,
    lw = 2,
    linestyle = :solid,
)
plot!(
    p1,
    t_range,
    pert_x,
    label = "Prey (perturbed)",
    color = :steelblue,
    lw = 2,
    linestyle = :dash,
    alpha = 0.6,
)
plot!(
    p1,
    t_range,
    pert_y,
    label = "Predator (perturbed)",
    color = :firebrick,
    lw = 2,
    linestyle = :dash,
    alpha = 0.6,
)

plot!(
    p1,
    xlabel = "Time",
    ylabel = "Population",
    title = "Lotka-Volterra Trajectories\n(solid = true params, dashed = perturbed)",
    legend = :topright,
    grid = true,
    size = (800, 400),
)

# ── Plot 2: Scaling ──────────────────────────────────────────────────────────

n_params_axis = 2 .* collect(species_counts)   # x-axis: total number of parameters

p2 = plot(
    n_params_axis,
    forward_times,
    label = "Forward (Dual)",
    color = :tomato,
    lw = 2.5,
    marker = :circle,
    markersize = 5,
)
plot!(
    p2,
    n_params_axis,
    reverse_times,
    label = "Reverse (Tape)",
    color = :mediumblue,
    lw = 2.5,
    marker = :square,
    markersize = 5,
)
plot!(
    p2,
    n_params_axis,
    zygote_times,
    label = "Zygote",
    color = :green,
    lw = 2.5,
    marker = :diamond,
    markersize = 5,
)
plot!(
    p2,
    n_params_axis,
    enzyme_times,
    label = "Enzyme",
    color = :purple,
    lw = 2.5,
    marker = :star5,
    markersize = 5,
)

plot!(
    p2,
    xlabel = "Number of parameters (n)",
    ylabel = "Median time (ms)",
    title = "Gradient Computation Time vs Number of Parameters\n(Lotka-Volterra, 400 Euler steps)",
    legend = :topleft,
    grid = true,
    size = (800, 450),
)

# ── Plot 3: Bar chart for 6-parameter case ───────────────────────────────────

med_times_ms = map(methods) do name
    median(timing_results[name]).time / 1e6
end

p3 = bar(
    methods,
    med_times_ms,
    color = [:tomato, :mediumblue, :green, :purple],
    xlabel = "Method",
    ylabel = "Median time (ms)",
    title = "Gradient Computation Time — 6-Parameter LV\n(median over 200 samples)",
    legend = false,
    grid = true,
    size = (700, 400),
)

# ── Save all plots ───────────────────────────────────────────────────────────

savefig(p1, "plot_trajectory.png")
savefig(p2, "plot_scaling.png")
savefig(p3, "plot_timing_bar.png")

println()
println("  Saved: plot_trajectory.png")
println("  Saved: plot_scaling.png")
println("  Saved: plot_timing_bar.png")
println()
println("=" ^ 65)
println("  Benchmark complete.")
println("=" ^ 65)
