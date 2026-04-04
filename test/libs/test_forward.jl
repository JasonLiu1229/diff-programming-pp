using Test

include("../../src/libs/forward.jl")

# ── Helpers ──────────────────────────────────────────────────────────────────
numerical_grad(f, x; ε=1e-5) = (f(x + ε) - f(x)) / ε


# ── Test functions ────────────────────────────────────────────────────────────

# f(x) = x*x + sin(x),  f'(x) = 2x + cos(x)
f(x) = x * x + sin(x)

# Falling ball simulation — plain Julia, no changes needed for Dual
function ball(v0)
    h = 0.0; v = v0
    for _ in 1:100
        h = h + v * 0.01
        v = v - 9.81 * 0.01
    end
    return h
end


# ── Tests ─────────────────────────────────────────────────────────────────────

@testset "Forward-mode AD (Dual numbers)" begin

    @testset "f(x) = x*x + sin(x)" begin
        x          = 1.0
        analytical = 2x + cos(x)   # f'(x) = 2x + cos(x)

        grad = dual_derivative(f, x)

        @test grad ≈ analytical              atol=1e-10
        @test grad ≈ numerical_grad(f, x)   rtol=1e-4
    end

    @testset "f'(x) holds at multiple input values" begin
        # The formula f'(x) = 2x + cos(x) should hold for any x
        for x in [0.0, 1.0, -1.0, 3.14]
            @test dual_derivative(f, x) ≈ 2x + cos(x)   atol=1e-10
        end
    end

    @testset "Falling ball: dh/dv0 = t_final = 1.0" begin
        # Analytical: nudging v0 by 1 m/s raises final height by t_final seconds
        grad = dual_derivative(ball, 20.0)

        @test grad ≈ 1.0                        atol=1e-10
        @test grad ≈ numerical_grad(ball, 20.0) rtol=1e-4
    end

end
