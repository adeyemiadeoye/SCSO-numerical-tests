using SparseArrays, LinearAlgebra
using SelfConcordantSmoothOptimization

using Random
using Distributions
function rr()
    rng = MersenneTwister(1234);
    return rng
end

Random.seed!(1234)

include("get_group.jl")

# using LIBSVMdata, JLD2, FileIO

glproblems = vcat(["sim_gl"])

# SPARSE GROUP LASSO PROBLEMS

function GroupLasso(data_name::String, m::Integer, n::Integer, grpsize, γ, μ; use_const_grpsize=false)
    A, y, x0, x_star, P = init_GroupLasso_models(data_name, m, n; grpsize=grpsize, use_const_grpsize=use_const_grpsize)
    grad_fx, hess_fx, jac_yx, grad_fy, hess_fy = get_derivative_fns_lasso(A, y)
    n = size(A,2)
    λmax = norm(A'*y,Inf)
    tau = P.tau = 0.9
    lambda_ = γ*λmax
    λ1 = tau*lambda_
    λ2 = (10-tau)*lambda_
    if n < 10000
        Lf = eigmax(A'*A)
    else
        Lf = maximum(sum(A.^2, dims=2))
    end
    f = x -> f_grouplasso(A, x, y)
    return Problem(A, y, x0, f, [λ1,λ2]; Lf=Lf, P=P, sol=x_star, out_fn=out_lasso, grad_fx=grad_fx, hess_fx=hess_fx, jac_yx=jac_yx, grad_fy=grad_fy, hess_fy=hess_fy, name=data_name)
end

function f_grouplasso(A, x::T, ys::B) where{T,B}
    m = size(ys, 1)
    return 0.5*sum(abs2.(out_lasso(A, x) - ys))
end

function out_lasso(A::AbstractArray{S,2}, x::T) where{S<:Real,T}
    return A*x
end

function init_GroupLasso_models(data_name, m, n; grpsize=100, use_const_grpsize=false)
    # m : number of samples
    # n : number of features

    if use_const_grpsize
        const_grpsize = grpsize
    else
        const_grpsize = nothing
    end
    grpNUM = Int(round(n/grpsize, digits=0))

    if data_name == "sim_gl"
        probs = get_probs(grpNUM)
        ind, grpSIZES = get_indgrpSIZES(grpNUM, n; const_grpsize=const_grpsize, probs=probs)
        A, y, x = generate_data(m, n, grpNUM, grpSIZES, ind; rho=0.5, p_active=0.1)
        _, G, _, _ = get_group(A, grpNUM; const_grpsize=const_grpsize, probs=probs, ind=ind)
    else
        Base.error("data_name not valid")
    end
    P = get_P(n, G, ind)
    x0 = randn(rr(), n)

	return A, y, x0, x, P
end

function generate_data(m, n, grpNUM, grpSIZES, ind; rho=0.5, p_active=0.1)
    """ Data generation process with Toplitz like correlated features:
        according to the paper "GAP Safe Screening Rules for Sparse-Group Lasso". See https://github.com/EugeneNdiaye/GAPSAFE_SGL
    """

    Random.seed!(1234)
    g_start = ind[1,:]
    g_end = ind[2,:]

    # 10% of groups are active
    gamma1 = ceil(Int, grpNUM * 0.1)
    selected_groups = rand(rr(), 1:grpNUM, gamma1)
    true_coef = zeros(n)

    for i in selected_groups
        begin_idx = g_start[i]
        end_idx = g_end[i]
        # p_active: proportion of features that are active
        gamma2 = ceil(Int, grpSIZES[i] * p_active)
        selected_features = rand(begin_idx:end_idx, gamma2)

        ns = length(selected_features)
        s = 2 .* rand(ns) .- 1
        u = rand(ns)
        true_coef[selected_features] .= sign.(s) .* (10 .* u .+ (1 .- u) .* 0.5)
    end

    vect = rho .^ collect(0:n-1)
    covar = make_toeplitz(vect)
    sn2 = 0.01

    A = Matrix(rand(rr(), MvNormal(zeros(n), covar), m)')
    y = A * true_coef + sn2 .* randn(rr(), m)

    return A, y, true_coef
end
function make_toeplitz(c)
    n = length(c)
    T = zeros(n, n)
    for i = 1:n
        for j = 1:n
            T[i, j] = c[abs(i-j)+1]
        end
    end
    return T
end

# get derivate functions for the for the linear regression problem
function get_derivative_fns_lasso(A::AbstractArray{Float64,2}, y::VectorOrBitVector{<:IntOrFloat})
    m = size(A, 1)
    grad_fx(x::Vector{Float64}) = A' * (out_lasso(A,x) - y)
    hess_fx(x::Vector{Float64}) = A' * A
    jac_yx(ŷ::Vector{Float64}) = A
    grad_fy(y_hat::Array{Float64}) = y_hat - y
    hess_fy(y_hat::Array{Float64}) = Diagonal((one.(y_hat)))
    return grad_fx, hess_fx, jac_yx, grad_fy, hess_fy
end