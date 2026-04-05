# A Dual number carries two things:
#   value → the actual number
#   deriv → the derivative riding alongside it
struct Dual
    value::Float64
    deriv::Float64
end

Base.:-(a::Dual) = Dual(-a.value, -a.deriv)
Base.:+(a::Dual) = Dual(a.value, a.deriv)

# ── Arithmetic on two Dual numbers ──────────────
Base.:+(a::Dual, b::Dual) = Dual(a.value + b.value, a.deriv + b.deriv)
Base.:-(a::Dual, b::Dual) = Dual(a.value - b.value, a.deriv - b.deriv)
Base.:*(a::Dual, b::Dual) = Dual(a.value * b.value, a.deriv * b.value + a.value * b.deriv)
Base.:/(a::Dual, b::Dual) =
    Dual(a.value / b.value, (a.deriv * b.value - a.value * b.deriv) / b.value^2)
Base.:^(a::Dual, n::Int) = Dual(a.value^n, n * a.value^(n-1) * a.deriv)

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
Base.sin(a::Dual) = Dual(sin(a.value), cos(a.value) * a.deriv)
Base.cos(a::Dual) = Dual(cos(a.value), -sin(a.value) * a.deriv)

# ── sqrt — chain rule: d/dx sqrt(x) = 1 / (2*sqrt(x)) ──
Base.sqrt(a::Dual) = Dual(sqrt(a.value), a.deriv / (2.0 * sqrt(a.value)))

# ── abs — chain rule: d/dx |x| = sign(x)  (undefined at 0, but fine in practice) ──
Base.abs(a::Dual) = Dual(abs(a.value), a.deriv * sign(a.value))

Base.zero(::Type{Dual}) = Dual(0.0, 0.0)
Base.zero(::Dual) = Dual(0.0, 0.0)
Base.one(::Type{Dual}) = Dual(1.0, 0.0)
Base.one(::Dual) = Dual(1.0, 0.0)

Base.convert(::Type{Dual}, x::Real) = Dual(Float64(x), 0.0)

function dual_derivative(f, x::Float64)
    result = f(Dual(x, 1.0))
    return result.deriv
end

function dual_gradient(f, params::Vector{Float64})
    n = length(params)
    grads = zeros(Float64, n)
    for i = 1:n
        dual_params = [Dual(params[j], j == i ? 1.0 : 0.0) for j = 1:n]
        result = f(dual_params...)
        grads[i] = result.deriv
    end
    return grads
end
