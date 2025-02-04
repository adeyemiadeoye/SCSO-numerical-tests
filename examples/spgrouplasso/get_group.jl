using Random
using Distributions
function rr()
    rng = MersenneTwister(1234);
    return rng
end

function get_group(A, grpNUM; const_grpsize=nothing, probs=nothing, ind=nothing)
    # A - data matrix
    # grpNUM - number of groups

    n = size(A, 2)
    if grpNUM > n
        Base.error("grpsize > number of variables")
    end
    G = collect(1:n)

    reorder = randperm(rr(), n)
    newA = A[:, reorder]

    # if a constant group size is not specified, use probs by default
    if const_grpsize === nothing && probs === nothing
        probs = 1 .+ 0.3 * sign.(randn(rr(), grpNUM)) .* rand(rr(), grpNUM)
        probs = probs / sum(probs)
        probs = cumsum(probs)
    end
    if ind === nothing
        ind, _ = get_indgrpSIZES(grpNUM, n; const_grpsize=const_grpsize, probs=probs)
    end

    return newA, G, ind, reorder
end

function get_probs(grpNUM)
    probs = 1 .+ 0.3 * sign.(randn(rr(), grpNUM)) .* rand(rr(), grpNUM)
    probs = probs / sum(probs)
    probs = cumsum(probs)
    return probs
end

function get_indgrpSIZES(grpNUM, n; const_grpsize=nothing, probs=nothing)
    # p: size of variable
    ind = zeros(Int, 3, grpNUM)
    for i in 1:grpNUM
        if i == 1
            if const_grpsize !== nothing
                tmp = round(Int, const_grpsize)
            else
                tmp = round(Int, probs[1] * n)
            end
            tmp = max(tmp, 1)
            ind[1, 1] = 1
            ind[2, 1] = tmp
            ind[3, 1] = isqrt(tmp)
        else
            ind[1, i] = ind[2, i - 1] + 1
            if const_grpsize !== nothing
                ind[2, i] = max(round(Int, ind[1, i]+const_grpsize-1), ind[1, i])
            else
                ind[2, i] = max(round(Int, probs[i] * n), ind[1, i])
            end
            ind[3, i] = isqrt(ind[2, i] - ind[1, i])
        end
    end
    grpSIZES = (ind[2, :] .- ind[1, :]) .+ 1
    return ind, grpSIZES
end
