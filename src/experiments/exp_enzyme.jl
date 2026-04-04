using Enzyme

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
const G = 9.81
const DT = 0.01
const STEPS = 100

function simulate_ball(v0)
    h = 0.0
    v = v0
    for _ = 1:STEPS
        h = h + v * DT
        v = v - G * DT
    end
    return h
end

# ─────────────────────────────────────────────────────────────────────────────
# DIFFERENTIATION
# ─────────────────────────────────────────────────────────────────────────────

v0 = 20.0
result = Enzyme.autodiff(Reverse, simulate_ball, Active, Active(v0))
grad = result[1][1]   # extract dh/dv0

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

t_final = STEPS * DT
analytical = t_final

ε = 1e-5
h_final = simulate_ball(v0)
numerical = (simulate_ball(v0 + ε) - simulate_ball(v0)) / ε

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

println("=" ^ 50)
println("  Enzyme AD — Falling Ball")
println("=" ^ 50)
println("  Initial velocity  v0   = $(v0) m/s")
println("  Simulated time         = $(t_final) s")
println("  Final height           = $(round(h_final, digits=4)) m")
println()
println("  Gradient dh/dv0:")
println("    Enzyme               = $(grad)")
println("    Analytical           = $(analytical)")
println("    Numerical (finite Δ) = $(round(numerical, digits=6))")
println()
println("  Error vs analytical    = $(abs(grad - analytical))")
println("  Matches numerical?     = $(isapprox(grad, numerical, rtol=1e-4))")
println("=" ^ 50)
