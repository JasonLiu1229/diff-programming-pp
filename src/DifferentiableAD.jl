module DifferentiableAD

include("libs/forward.jl")
include("libs/reverse.jl")

export dual_derivative   # from forward.jl
export gradient, forward, backward, new_slot!, taped_add,
       taped_sub, taped_mul, taped_div, taped_sin, taped_cos  # from reverse.jl

end
