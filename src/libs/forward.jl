# A Dual number carries two things:
#   value → the actual number
#   deriv → the derivative riding alongside it
struct Dual
    value::Float64
    deriv::Float64
end

# ── Arithmetic on two Dual numbers ──────────────
Base.:+(a::Dual, b::Dual) = Dual(a.value + b.value, a.deriv + b.deriv)
Base.:-(a::Dual, b::Dual) = Dual(a.value - b.value, a.deriv - b.deriv)
Base.:*(a::Dual, b::Dual) = Dual(a.value * b.value, a.deriv * b.value + a.value * b.deriv)
Base.:/(a::Dual, b::Dual) = Dual(a.value / b.value, (a.deriv * b.value - a.value * b.deriv) / b.value^2)
Base.:^(a::Dual, n::Int)  = Dual(a.value^n, n * a.value^(n-1) * a.deriv)

# ── Mixed arithmetic (Dual with plain numbers) ──
Base.:+(a::Dual, b::Real) = Dual(a.value + b, a.deriv)
Base.:+(a::Real, b::Dual) = Dual(a + b.value, b.deriv)
Base.:-(a::Dual, b::Real) = Dual(a.value - b, a.deriv)
Base.:-(a::Real, b::Dual) = Dual(a - b.value, -b.deriv)
Base.:*(a::Dual, b::Real) = Dual(a.value * b, a.deriv * b)
Base.:*(a::Real, b::Dual) = Dual(a * b.value, a * b.deriv)
Base.:/(a::Dual, b::Real) = Dual(a.value / b, a.deriv / b)
Base.:/(a::Real, b::Dual) = Dual(a / b.value, -a * b.deriv / b.value^2)

# ── Trig functions — chain rule ──────────────────
Base.sin(a::Dual) = Dual(sin(a.value),  cos(a.value) * a.deriv)
Base.cos(a::Dual) = Dual(cos(a.value), -sin(a.value) * a.deriv)


function dual_derivative(f, x::Float64)
    result = f(Dual(x, 1.0))
    return result.deriv
end
