using Test

include("../../src/libs/reverse.jl")

# ── Helpers ──────────────────────────────────────────────────────────────────
function numerical_grad(f_plain, x; ε=1e-5)
    (f_plain(x + ε) - f_plain(x)) / ε
end


# ── Test functions ────────────────────────────────────────────────────────────

# f(x) = x*x + sin(x),  f'(x) = 2x + cos(x)
# Written using taped_* primitives so the tape gets built during the forward pass.
function f_taped(tape, slots)
    x  = slots[1]
    xx = taped_mul(tape, x, x)    # x * x
    sx = taped_sin(tape, x)       # sin(x)
    taped_add(tape, xx, sx)       # x*x + sin(x) — return the output slot
end

# Plain version of the same function — used only for the numerical gradient check
f_plain(x) = x * x + sin(x)

# Falling ball simulation using taped primitives
function ball_taped(tape, slots)
    G, DT, STEPS = 9.81, 0.01, 100
    cur_v    = slots[1]
    cur_h    = new_slot!(tape, 0.0)
    dt_slot  = new_slot!(tape, DT)
    gdt_slot = new_slot!(tape, G * DT)
    for _ in 1:STEPS
        v_dt  = taped_mul(tape, cur_v, dt_slot)
        cur_h = taped_add(tape, cur_h, v_dt)
        cur_v = taped_sub(tape, cur_v, gdt_slot)
    end
    return cur_h   # output slot
end


# ── Tests ─────────────────────────────────────────────────────────────────────

@testset "Reverse-mode AD (tape-based)" begin

    @testset "f(x) = x*x + sin(x)" begin
        x          = 1.0
        analytical = 2x + cos(x)   # f'(x) = 2x + cos(x)

        grad = gradient(f_taped, [x])[1]

        @test grad ≈ analytical                  atol=1e-10
        @test grad ≈ numerical_grad(f_plain, x)  rtol=1e-4
    end

    @testset "f'(x) holds at multiple input values" begin
        for x in [0.0, 1.0, -1.0, 3.14]
            @test gradient(f_taped, [x])[1] ≈ 2x + cos(x)   atol=1e-10
        end
    end

    @testset "Tape is non-empty after forward pass" begin
        # Sanity check: the tape should have recorded operations
        loss_slot, tape, _ = forward(f_taped, [1.0])
        @test length(tape.entries) > 0
        @test length(tape.values)  > 0
    end

    @testset "Gradient accumulation: slot used by two operations" begin
        f_sq(tape, slots) = taped_mul(tape, slots[1], slots[1])
        @test gradient(f_sq, [3.0])[1] ≈ 6.0   atol=1e-10
    end

    @testset "Falling ball: dh/dv0 = t_final = 1.0" begin
        grad = gradient(ball_taped, [20.0])[1]
        @test grad ≈ 1.0   atol=1e-10
    end

end
