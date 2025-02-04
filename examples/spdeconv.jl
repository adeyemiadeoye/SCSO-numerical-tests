using SparseArrays, DSP

using Random
using Distributions
function rr()
    rng = MersenneTwister(1234);
    return rng
end


# SPARSE DECONVOLUTION PROBLEM

function SpDeconv(model_name::String, N::Integer, λ)
    A, B, H, y, noise, x = init_SpDeconv_models(N)
    grad_fx, hess_fx, jac_yx, grad_fy, hess_fy = get_derivative_fns_deconv(A, B, y)
    AB = A\B
    Lf = eigmax(1/N * (AB'*AB))
    f = x -> f_deconv(H, x, y)
    return Problem(H, y, y, f, λ; Lf=Lf, sol=x, out_fn=out_deconv, grad_fx=grad_fx, hess_fx=hess_fx, jac_yx=jac_yx, grad_fy=grad_fy, hess_fy=hess_fy, name=model_name)
end

function f_deconv(Hs, x::T, ys::B) where{T,B}
    n = size(ys, 1)
    return 0.5/n*sum(abs2.(out_deconv(Hs, x) - ys))
end

function out_deconv(Hs::Function, x::T) where{T}
    return Hs(x)
end

function init_SpDeconv_models(N)
    # N : signal length

    # number of diracs
    p = Int(round(N*.03))
    # location of the diracs
    sel = randperm(rr(), N)
    sel = sel[1:p]
    # signal
    x = zeros(N)
    x[sel] .= 1
    x = x .* sign.(randn(rr(), N)) .* (1 .- 0.5.*rand(rr(), N))

    b = [1, 0.8] # Define filter
    r = 0.9
    om = 0.95
    a = [1, -2*r*cos(om), r^2]
    hx = filt(b,a,x)
    sigma = 0.2
    noise = sigma*randn(rr(), N)
    y = hx + noise
    Nb = length(b)
    Na = length(a)
    b1 = reduce(hcat,[ones(N,1),reduce(hcat, [repeat(b[j,:],N) for j in 2:Nb])[:,:]])
    bb = [i => b1[:,j] for (i,j) in zip(collect(0:-1:1-Nb), axes(b1,2))]
    B = spdiagm(bb...)[1:N,1:N]
    a1 = reduce(hcat,[ones(N,1),reduce(hcat, [repeat(a[j,:],N) for j in 2:Na])[:,:]])
    aa = [i => a1[:,j] for (i,j) in zip(collect(0:-1:1-Na), axes(a1,2))]
    A = spdiagm(aa...)[1:N,1:N]
    H(x) = A\(B*x)

	return A, B, H, y, noise, x
end

# get derivate functions for the for the linear regression problem
function get_derivative_fns_deconv(Ah::AbstractArray{Float64,2}, Bh::AbstractArray{Float64,2}, y::VectorOrBitVector{<:IntOrFloat})
    A = Ah \ Bh
    n = size(A, 1)
    grad_fx(x::Vector{Float64}) = 2/n * A' * (A * x - y)
    hess_fx(x::Vector{Float64}) = 2/n * A' * A
    jac_yx(ŷ::Vector{Float64}) = A
    grad_fy(y_hat::Array{Float64}) = 2/n * (y_hat - y)
    hess_fy(y_hat::Array{Float64}) = Diagonal((2/n * one.(y_hat)))
    return grad_fx, hess_fx, jac_yx, grad_fy, hess_fy
end