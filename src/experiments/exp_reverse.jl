include("../libs/reverse.jl")

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

const G = 9.81
const DT = 0.01
const STEPS = 100

function simulate_ball(tape, slots)
    cur_v = slots[1]                      # v0 — the input we differentiate w.r.t.
    cur_h = new_slot!(tape, 0.0)          # initial height — constant, not differentiated
    dt_slot = new_slot!(tape, DT)           # constant
    gdt_slot = new_slot!(tape, G * DT)       # constant — pre-multiply g*dt once

    for _ = 1:STEPS
        v_dt = taped_mul(tape, cur_v, dt_slot)   # v * dt
        cur_h = taped_add(tape, cur_h, v_dt)      # h = h + v*dt
        cur_v = taped_sub(tape, cur_v, gdt_slot)  # v = v - g*dt
    end

    return cur_h   # return the output slot — this is what we differentiate
end

# ─────────────────────────────────────────────────────────────────────────────
# DIFFERENTIATION
# ─────────────────────────────────────────────────────────────────────────────
#
# gradient() runs two phases internally:
#   1. forward()  — calls simulate_ball, builds the tape, records all operations
#   2. backward() — walks the tape in reverse, propagates gradients back to v0



v0 = 20.0
grad = gradient(simulate_ball, [v0])[1]

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

t_final = STEPS * DT    # 1.0 second
analytical = t_final       # dh/dv0 = t_final — same reasoning as forward experiment

# Numerical check — uses plain Julia simulation, completely independent of the tape
function simulate_plain(v0)
    h = 0.0;
    v = v0
    for _ = 1:STEPS
        h = h + v * DT
        v = v - G * DT
    end
    return h
end

ε = 1e-5
h_final = simulate_plain(v0)
numerical = (simulate_plain(v0 + ε) - simulate_plain(v0)) / ε

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

println("=" ^ 50)
println("  Reverse-mode AD — Falling Ball")
println("=" ^ 50)
println("  Initial velocity  v0   = $(v0) m/s")
println("  Simulated time         = $(t_final) s")
println("  Final height           = $(round(h_final, digits=4)) m")
println()
println("  Gradient dh/dv0:")
println("    Reverse-mode (tape)  = $(grad)")
println("    Analytical           = $(analytical)")
println("    Numerical (finite Δ) = $(round(numerical, digits=6))")
println()
println("  Error vs analytical    = $(abs(grad - analytical))")
println("  Matches numerical?     = $(isapprox(grad, numerical, rtol=1e-4))")
println("=" ^ 50)
