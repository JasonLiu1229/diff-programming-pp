# reverse.jl  —  Tape-based Reverse-Mode Automatic Differentiation
# The core idea:
#   Forward pass  →  run the computation normally, but secretly log every
#                    operation and its input values onto a "tape"
#   Backward pass →  walk that tape in reverse, applying the chain rule
#                    at each step to propagate gradients back to the inputs


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

# One entry on the tape — a single primitive operation that was recorded.
#
#   op     → which operation was performed (:add, :mul, :sin, :cos)
#   inputs → slot indices of the input values
#   output → slot index where the result was stored
#   vals   → the actual numeric input values at the time of the operation
#             (we need these later because local derivatives often depend
#              on the value — e.g. d(sin(a))/da = cos(a), so we need 'a')
struct TapeEntry
    op     :: Symbol
    inputs :: Vector{Int}
    output :: Int
    vals   :: Vector{Float64}
end

# The tape itself: a log of operations and a flat array of all values.
# Every intermediate result gets a numbered "slot" in the values array.
# Think of slots as numbered registers in a tiny virtual machine.
mutable struct Tape
    entries :: Vector{TapeEntry}
    values  :: Vector{Float64}
end

Tape() = Tape(TapeEntry[], Float64[])


# ─────────────────────────────────────────────────────────────────────────────
# TAPE OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

# Allocate a new slot, store a value in it, and return its index.
# Used for both input variables and constants.
function new_slot!(tape::Tape, val::Float64)::Int
    push!(tape.values, val)
    return length(tape.values)
end

# Record a primitive operation onto the tape and return the output slot.
function record!(tape::Tape, op::Symbol, input_slots::Vector{Int}, result::Float64)::Int
    out_slot   = new_slot!(tape, result)
    input_vals = [tape.values[i] for i in input_slots]
    push!(tape.entries, TapeEntry(op, input_slots, out_slot, input_vals))
    return out_slot
end

taped_add(t, a, b) = record!(t, :add, [a, b], t.values[a] + t.values[b])
taped_sub(t, a, b) = record!(t, :sub, [a, b], t.values[a] - t.values[b])
taped_mul(t, a, b) = record!(t, :mul, [a, b], t.values[a] * t.values[b])
taped_div(t, a, b) = record!(t, :div, [a, b], t.values[a] / t.values[b])
taped_sin(t, a)    = record!(t, :sin, [a],    sin(t.values[a]))
taped_cos(t, a)    = record!(t, :cos, [a],    cos(t.values[a]))


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

function forward(f::Function, inputs::Vector{Float64})
    tape = Tape()

    # Allocate one slot per input value and collect their indices.
    # The caller gets these back so they can look up dL/d(input_i) in grads.
    input_slots = [new_slot!(tape, v) for v in inputs]

    # The user-defined function builds the computation on the tape
    # and returns the slot index of the final output (the "loss").
    loss_slot = f(tape, input_slots)

    loss = tape.values[loss_slot]
    return loss_slot, tape, input_slots
end


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

function backward(tape::Tape, loss_slot::Int)
    grads = zeros(Float64, length(tape.values))

    # Seed: the loss is 100% sensitive to itself — always start with 1.0
    grads[loss_slot] = 1.0

    for entry in reverse(tape.entries)
        g_out = grads[entry.output]   # gradient flowing into this op's output

        if entry.op == :add
            grads[entry.inputs[1]] += g_out
            grads[entry.inputs[2]] += g_out

        elseif entry.op == :sub
            grads[entry.inputs[1]] += g_out
            grads[entry.inputs[2]] -= g_out

        elseif entry.op == :mul
            # out = a * b  →  d/da = b,  d/db = a
            a_val, b_val = entry.vals[1], entry.vals[2]
            grads[entry.inputs[1]] += g_out * b_val
            grads[entry.inputs[2]] += g_out * a_val

        elseif entry.op == :div
            # out = a / b  →  d/da = 1/b,  d/db = -a/b²
            a_val, b_val = entry.vals[1], entry.vals[2]
            grads[entry.inputs[1]] += g_out / b_val
            grads[entry.inputs[2]] -= g_out * a_val / b_val^2

        elseif entry.op == :sin
            # out = sin(a)  →  d/da = cos(a)
            a_val = entry.vals[1]
            grads[entry.inputs[1]] += g_out * cos(a_val)

        elseif entry.op == :cos
            # out = cos(a)  →  d/da = -sin(a)
            a_val = entry.vals[1]
            grads[entry.inputs[1]] += g_out * (-sin(a_val))
        end
    end

    return grads
end


# ─────────────────────────────────────────────────────────────────────────────
# GENERAL GRADIENT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
function gradient(f::Function, inputs::Vector{Float64})
    loss_slot, tape, input_slots = forward(f, inputs)
    grads = backward(tape, loss_slot)
    return [grads[s] for s in input_slots]   # one gradient per input, in order
end
