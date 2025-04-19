"""
   data utility functions.
        modified https://github.com/adeyemiadeoye/ggn-score-nn/blob/main/experiments/utils/data-utils.jl
"""

using MLUtils
using MLDatasets
using DelimitedFiles
using LIBSVMdata
using Flux
using Flux: onehotbatch, onecold
using Random
using Distributions
function rr()
    rng = MersenneTwister(1234)
    return rng
end

Random.seed!(1234)

function permute_inp_data(inp, permutation)
    return inp[permutation]
end
function create_permuted_datasets(Xtrain, Xtest, permutation)
    permuted_Xtrain = [permute_inp_data(x, permutation) for x in eachslice(Xtrain, dims=ndims(Xtrain))]
    permuted_Xtest = [permute_inp_data(x, permutation) for x in eachslice(Xtest, dims=ndims(Xtest))]
    permuted_Xtrain = reshape(reduce(hcat,permuted_Xtrain), size(Xtrain))
    permuted_Xtest = reshape(reduce(hcat,permuted_Xtest), size(Xtest))
    return permuted_Xtrain, permuted_Xtest
end
function get_data(;train_dataset=nothing, test_dataset=nothing, raw_data=false, data_name="mnist", test_available=true, perm_mode=false, shuff_mode=false, seq_mode=false)
    if all(x->x!==nothing,(train_dataset, test_dataset))
        train_inps, train_labels = train_dataset[:]
        test_inps, test_labels = test_dataset[:]
    else
        Xtr, train_labels = LIBSVMdata.load_dataset(data_name, dense=true, replace=false, verbose=false)
        if test_available
            Xte, test_labels = LIBSVMdata.load_dataset(data_name*".t", dense=true, replace=false, verbose=false)
        else
            train_ratio = 0.8
            train_size = Int(round(train_ratio * size(Xtr, 1)))
            Xte = Xtr[train_size+1:end, :]
            test_labels = train_labels[train_size+1:end]
            Xtr = Xtr[1:train_size, :]
            train_labels = train_labels[1:train_size]
        end
        train_inps = Matrix(Xtr')
        test_inps = Matrix(Xte')
    end
    train_classes = sort(unique(train_labels))
    test_classes = sort(unique(test_labels))
    if raw_data
        return train_inps, train_labels, test_inps, test_labels, train_classes, test_classes
    end
    if perm_mode
        pixel_indices = shuffle(rr(), 1:prod(size(train_inps)[1:end-1]))
        train_inps, test_inps = create_permuted_datasets(train_inps, test_inps, pixel_indices)
    end
    
    Xtrain = Flux.flatten(train_inps)
    Xtest = Flux.flatten(test_inps)
    ytrain = onehotbatch(train_labels, train_classes)
    ytest = onehotbatch(test_labels, test_classes)
    if shuff_mode # optionally shuffle the training data
        shuff = randperm(rr(), size(Xtrain, 2))
        Xtrain = Xtrain[:, shuff]
        ytrain = ytrain[:, shuff]
    end
    if seq_mode
        Xtrain = [Xtrain[:,i] for i in axes(Xtrain,2)]
        Xtest = [Xtest[:,i] for i in axes(Xtest,2)]
    else
        Xtrain, ytrain, Xtest, ytest = Matrix{Float64}(Xtrain'), Matrix{Float64}(ytrain'), Matrix{Float64}(Xtest'), Matrix{Float64}(ytest')
    end

    return Xtrain, ytrain, Xtest, ytest, train_classes, test_classes
end


normalize_data(x) = (x .- 0.1307f0) ./ 0.3081f0

function get_jl_dataset(;train_dataset=nothing, test_dataset=nothing, data_name="w1a", raw_data=false, test_available=true, normalize_train=false, for_fed=false)
    # Load data
    if raw_data
        train_x, train_y, test_x, test_y, train_classes, test_classes = get_data(train_dataset=train_dataset, test_dataset=test_dataset, raw_data=true, data_name=data_name, test_available=test_available)
        train_x = permutedims(train_x, (3, 1, 2))
        test_x = permutedims(test_x, (3, 1, 2))
    else
        train_x, train_y, test_x, test_y, train_classes, test_classes = get_data(train_dataset=train_dataset, test_dataset=test_dataset, raw_data=false, data_name=data_name, test_available=test_available)
        if for_fed
            train_y = onecold(train_y', train_classes)
        end
    end

    # normalize training data?
    if normalize_train
        train_x = normalize_data(train_x)
    end

    return train_x, train_y, test_x, test_y, train_classes, test_classes
end