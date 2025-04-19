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
# e.g, provide A and y as arguments to Problem constructor below
# (representing the training inputs and labels) with same dimensions as the test inputs
# and add training accuracy to metrics (uncomment the line below)
function train_accuracy(SCSOProblem, θ)
    Φ_model(X) = SCSOProblem.out_fn(X, θ)
    accuracy(Φ_model, SCSOProblem.A, SCSOProblem.y')
end
# metrics["train_accuracy"] = train_accuracy # add train_accuracy to metrics

function accuracy(SCSOProblem, X, y)
    ŷ = SCSOProblem(X)
    correct = onecold(ŷ) .== onecold(y)
    round(100 * mean(correct); digits=2)
end

# (you can use your own dataset)
train_x, train_y, test_x, test_y, _, _ = get_jl_dataset(data_name="w1a", for_fed=true)

# The get_fed_dataset utility function is based on the python package fedartml,
# loaded internally via PythonCall (hence, ensure you have the package installed in your python environment)
num_clients = 4
clients_data, client_idxs, miss_class_per_node, distances = get_fed_dataset(train_x, train_y, num_clients; method="percent_noniid", percent_noniid=60, with_class_completion=true)

Random.seed!(1234)
global_model = Chain(
    Dense(size(clients_data[1].features,2), 128, identity),
    Dense(128, size(clients_data[1].targets,2)),
    softmax
)
global_model = fmap(f64, global_model) # Float32 to Float64 conversion

# define the loss function
f(ŷ, y) = Flux.logitcrossentropy(ŷ, y)

# define the regularization/smoothing functions
reg_name = "l1"
μ = 1.0
hμ = PHuberSmootherL1L2(μ)
λ = 1e-4

Random.seed!(1234) # for reproducibility

# define the problem
fed_problem = Problem(clients_data, global_model, f, λ; Atest=test_x, ytest=test_y)

# define the local algorithm
local_algo = ProxGGNSCORE(use_prox=false)

# begin training
result = iterate!(local_algo, fed_problem, reg_name, hμ; batch_size=8, comm_rounds=10, local_max_iter=500, α=1.0, metrics=metrics, shuffle_batch=true, verbose=1)