include("utils/data-utils.jl")

using SelfConcordantSmoothOptimization

# optional accuracy metrics to compute:
# Any accuracy function provided must take an SCSOProblem and the optimization variable θ
# as arguments. The function must return a scalar value.
function test_accuracy(SCSOProblem, θ)
    Φ_model(X) = SCSOProblem.out_fn(X, θ)
    accuracy(Φ_model, SCSOProblem.Atest, SCSOProblem.ytest')
end
metrics = Dict() # define a Dict and pass the metric name and the metric function as key-value pairs
metrics["test_accuracy"] = test_accuracy # add test_accuracy to metrics

# you can add any other metrics you want to compute
function train_accuracy(SCSOProblem, θ)
    Φ_model(X) = SCSOProblem.out_fn(X, θ)
    accuracy(Φ_model, SCSOProblem.A, SCSOProblem.y')
end
metrics["train_accuracy"] = train_accuracy

function accuracy(SCSOProblem, X, y)
    ŷ = SCSOProblem(X)
    correct = onecold(ŷ) .== onecold(y)
    round(100 * mean(correct); digits=2)
end

# (you can use your own dataset)
train_x, train_y, test_x, test_y, _, _ = get_jl_dataset(data_name="w1a")


Random.seed!(1234)
nn_model = Chain(
    Dense(size(train_x,2), 128, identity),
    Dense(128, size(train_y,2)),
    softmax
)
nn_model = fmap(f64, nn_model) # Float32 to Float64 conversion
x0 = Flux.destructure(nn_model)[1]

# define the loss function
f(ŷ, y) = Flux.logitcrossentropy(ŷ, y)

# define the regularization/smoothing functions
reg_name = "l1"
μ = 1.0
hμ = PHuberSmootherL1L2(μ)
λ = 1e-4

Random.seed!(1234) # for reproducibility

# define the problem
nn_problem = Problem(train_x, train_y, x0, f, λ; out_fn=nn_model, Atest=test_x, ytest=test_y)

# define the local algorithm
optimizer = ProxGGNSCORE(use_prox=false)
max_epoch = 10

# begin training
result = iterate!(optimizer, nn_problem, reg_name, hμ; batch_size=8, max_epoch=max_epoch, α=1e-2, metrics=metrics, shuffle_batch=true, verbose=2)