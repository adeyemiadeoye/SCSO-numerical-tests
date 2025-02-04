include("exp-utils-paper.jl");
using NPZ

# uncomment the desired line to run


##### RUN INDIVIDUAL EXAMPLE #####
# adjust the parameters according to paper

## sparse logistic regression
@time model, _, sol = solve!("sim_log", "prox-ggnscore"; m=100, n=4500, λ=2.0e-1, μ=1, max_iter=2000, reg_name="l1");

## sparse deconvolution
# @time model, _, sol = solve!("deconv", "prox-grad"; m=1024, n=200, λ=1e-3, μ=5e-2, max_iter=5000, reg_name="l1", ss_type=1, x_tol=1e-6, f_tol=1e-6, verbose=2, α=1);


## sparse group lasso
# m, n = 500, 2000
# problem = "sim_gl"
# method = "prox-ggnscore"
# if method in ["prox-ggnscore", "prox-newtonscore", "prox-bfgsscore"]
#     α = 1
#     λ = 1e-8
# else
#     # let other algorithms use L for α
#     # better results with λ = 1e-7
#     α = nothing
#     λ = 1e-7
# end
# @time model, _, sol = solve!(problem, method; m=m, n=n, λ=λ, μ=1.6, max_iter=100000, reg_name="gl", grpsize=100, use_const_grpsize=true, ss_type=1, verbose=2, x_tol=1e-9, α=α);



##### RUN ALL THE EXAMPLES AND PLOT #####
# uncomment to run

# results = RUNPaperExperiments();

# plot_allresults(results);

# RUNPaperExperiments_α(200, 3000);

# result, _ = RUNPaperExperiments_SGL(1000, 10000);

# plot_performance_profile(n_runs=20);





;