using SelfConcordantSmoothOptimization

using DataStructures
using StatsBase

using Plots; pyplot()
using Formatting
using BenchmarkProfiles

include("splogl1.jl")
include("spdeconv.jl")
include("spgrouplasso/spgrouplasso.jl")
include("load_bcd.jl")
include("load_ssnal.jl")

markershape = [:dtriangle :rtriangle :rect :diamond :hexagon :xcross :utriangle :ltriangle :pentagon :heptagon :octagon :star4 :star6 :star7 :star8]

_plot_args = Dict{Symbol, Any}(
    :background => :white,
    :framestyle => :box,
    :grid => true,
    :gridalpha => 0.2,
    :linewidth => 2,
    :tickfontsize => 8,
    :minorgrid => true,
    :gridlinewidth => 0.7,
    :minorgridalpha => 0.06,
    :palette => :tab10,
    :background_color_legend => RGBA(1.0,1.0,1.0,0.8),
    :size => (500, 400)
)

default(; _plot_args...)

# CUSTOM MARKERS
@recipe function f(::Type{Val{:samplemarkers}}, x, y, z)
    n = length(y)
    if n > 20
        step = Int(round(n/10))
    else
        step = 1
    end
    sx, sy = x[1:step:n], y[1:step:n]
    # add an empty series with the correct type for legend markers
    @series begin
        seriestype := :path
        markershape --> :auto
        x := []
        y := []
    end
    # add a series for the line
    @series begin
        primary := false # no legend entry
        markershape := :none # ensure no markers
        seriestype := :path
        seriescolor := get(plotattributes, :seriescolor, :auto)
        x := x
        y := y
    end
    # return  a series for the sampled markers
    primary := false
    seriestype := :scatter
    markersize := 5
    
    markershape --> :auto
    x := sx
    y := sy
end

# MAIN FUNCTIONS
function solve!(model_name::String, method_name::String;
                m=2000, n=300, λ=2.0, ss_type=1, μ=2.0, max_iter=240, x_tol=1e-10, f_tol=1e-10, reg_name="l1", log_reg_problems=nothing, α=1.0, lb=-1.0, ub=1.0, grpsize=100, use_const_grpsize=false, verbose=0, seed=1234)

    #   model_name : {"sim_log", "deconv"}
    #   method_name : {"prox-ggnscore", "prox-grad", "prox-newtonscore", "prox-owlqn"}
    #   m : number of samples for synthetic data
    #   n : number of features for synthetic data
    #   λ : penalty/regularization parameter
    #   ss_type : how to select step length; 1 => use 1/Lipschitz_constant, 2=>use BB step-size, 3=>backtracking line-search
    #   μ : smoothness prameter for SCORE methods

    #   max_iter : maximum number of iterations
    #   reg_name : regularization function to use -> e.g. "l1", "l2"
    if log_reg_problems === nothing
        log_reg_problems = vcat(keys(data_dict)...,"sim_log")
    end
    extra_metrics = false
    if model_name in log_reg_problems # sparse logistic regression problem
        if reg_name == "l1"
            model = SpLogL1(model_name, m, n, λ; seed=seed)
            g = NormL1(λ) # g function for benchmark algos
        else
            Base.error("Please choose reg_name='l1'.")
        end
    elseif model_name == "deconv"
        extra_metrics = true
        if reg_name == "l1"
            model_name = "deconv-l1"
            g = NormL1(λ) # g function for benchmark algos
        elseif reg_name == "l2"
            model_name = "deconv-l2"
            g = NormL2(λ) # g function for benchmark algos
        else
            Base.error("Please choose reg_name in {'l1', 'l2'}.")
        end
        model = SpDeconv(model_name, m, λ)
    elseif model_name in glproblems
        reg_name = "gl"
        γ = λ
        model = GroupLasso(model_name, m, n, grpsize, γ, μ; use_const_grpsize=use_const_grpsize)
    else
        Base.error("Please choose model_name in $(vcat(log_reg_problems, "deconv", "boxqp")).")
    end

    if method_name in ["panoc", "zerofpr", "f-prox-grad"]
        x0 = model.x0
        if model.L === nothing
            Base.error("Please, provide L for the becnhmark algorithms.")
        end
        L = model.L
        f = model.f
        if method_name == "panoc"
            method = BMPANOC()
        elseif method_name == "zerofpr"
            method = BMZeroFPR()
        else
            method = BMFastProxGrad()
        end
        iter_states = get_iter_states(method, x0, f, g, L)

        solution = iterate_bmark(model, iter_states, reg_name; x_tol=x_tol, f_tol=f_tol, extra_metrics=extra_metrics)
    else
        if reg_name in ["l1", "l2"]
            hμ = PHuberSmootherL1L2(μ)
        elseif reg_name == "gl"
            hμ = PHuberSmootherGL(μ, model)
        end
        if method_name == "prox-ggnscore"
            method = ProxGGNSCORE(ss_type=ss_type)
        elseif method_name == "prox-grad"
            α = nothing
            hμ = NoSmooth(1.0)
            method = ProxGradient(ss_type=ss_type)
        elseif  method_name == "prox-newtonscore"
            method = ProxNSCORE(ss_type=ss_type)
        elseif  method_name == "prox-bfgsscore"
            method = ProxQNSCORE(ss_type=ss_type)
        elseif  method_name == "prox-owlqn"
            α = nothing
            hμ = NoSmooth(1.0)
            method = OWLQN(ss_type=ss_type, m=20)
        else
            Base.error("Please choose method_name in {'prox-newtonscore', 'prox-ggnscore', 'panoc', 'zerofpr', 'prox-owlqn', 'prox-grad', 'f-prox-grad', 'prox-bfgsscore'}.")
        end

        solution = iterate!(method, model, reg_name, hμ; α=α, max_iter=max_iter, extra_metrics=extra_metrics, x_tol=x_tol, f_tol=f_tol, verbose=verbose)
    end

    return model, method.label, solution

end

function get_density()
    d = Dict()
    for data_name in keys(data_dict)
        m, n = data_dict[data_name][1]
        ext = data_dict[data_name][2] # file extension
        if data_name == "gisette"
            dataset_path = "data/"*data_name*"_train"
            A = readdlm(dataset_path*".data")
            y = vec(readdlm(dataset_path*".labels"))
        else
            dataset_path = "data/"*data_name*ext
            A, y = fetch_data(dataset_path, m, n)
        end
        sA = size(A)
        sA = sA[1]*sA[2]
        nz = nnz(sparse(A))
        di = nz/sA
        d[data_name] = round(di, digits=2)
    end
    return d
end

function RUNPaperExperiments()
    problems_list = ["sim_log"]
    log_reg_problems = problems_list[1:end]


    methods_list = ["prox-newtonscore", "prox-ggnscore", "panoc", "zerofpr", "prox-owlqn", "prox-grad", "f-prox-grad"]
    reg_list = ["l1", "l2"]

    results = []
    n = 4500
    for model_name in problems_list
        for method_name in methods_list
            if model_name in log_reg_problems
                m = 100
                reg_name = reg_list[1]
                if model_name == "sim_log"
                    λ = 2.0e-1
                    μ = 1.0
                else
                    λ = 1.0
                    μ = 1.0
                end
                max_iter = 2000
                @info "Now solving problem $model_name using $method_name..."
                model, method_label, sol = solve!(model_name, method_name; m=m, n=n, λ=λ, μ=μ, max_iter=max_iter, reg_name=reg_name, log_reg_problems=log_reg_problems)
                push!(results, (model, method_name, method_label, sol))
            else
                x_tol = 1e-6
                f_tol = 1e-6
                if method_name in ["zerofpr", "panoc", "f-prox-grad", "prox-ggnscore", "prox-grad", "prox-newtonscore"]
                    m = 1024
                    λ = 1e-3
                    μ = 5e-2
                    max_iter = 5000
                    for reg_name in reg_list
                        @info "Now solving problem $model_name using $method_name, reg: $reg_name..."
                        model, method_label, sol = solve!(model_name, method_name; m=m, n=n, λ=λ, μ=μ, max_iter=max_iter, reg_name=reg_name, log_reg_problems=log_reg_problems, x_tol=x_tol, f_tol=f_tol)
                        push!(results, (model, method_name, method_label, sol))
                    end
                end
            end
        end
    end

    return results

end

function RUNPaperExperiments_α(m, n)
    file_dir = pwd() * "/figures/"
    yticks = [1e2,1e0,1e-2,1e-4,1e-6,1e-8]

    problems_list = ["sim_log"]
    methods_list = ["prox-ggnscore", "prox-newtonscore"]
    alpha_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    reg_name = "l1"
    
    max_iter = 2000
    for model_name in problems_list
        if model_name == "sim_log"
            pre_dir = "$(m)_$(n)_alpha"
            alias_name = "synthetic dataset: \$m=$m\$, \$n=$n\$"
            λ = 2e-1
            μ = 1.
        else
            pre_dir = "alpha"
            alias_name = model_name * " dataset: \$m=$(data_dict[model_name][1][1])\$, \$n=$(data_dict[model_name][1][2])\$"
            λ = 1.
            μ = 1.
        end

        for method_name in methods_list
            frel = []
            ts = []
            labels = []
            f_rel_ns = []
            for α in alpha_list
                @info "Now solving problem $model_name using $method_name with α = $α..."
                model, method_label, sol = solve!(model_name, method_name; m=m, n=n, λ=λ, μ=μ, max_iter=max_iter, reg_name=reg_name, log_reg_problems=problems_list, α=α)
                objrel = sol.objrel
                push!(f_rel_ns, length(objrel))
                push!(frel, objrel)
                push!(ts, sol.times)
                if α !== nothing
                    use_this_label = method_label*": \$\\alpha = $α\$"
                else
                    use_this_label = method_label*": \$\\alpha = \\max\\{1/L,1\\}\$"
                end
                push!(labels, use_this_label)
            end
            n = length(labels)
            labels = reshape(labels, (1,n))
            max_n = maximum(f_rel_ns)
            max_n = Int(round(max_n/min(max_n,5), digits=0))
            frelplot_alpha = plot(frel, label=labels, ylabel="\$\\|\\mathcal{L}_k-\\mathcal{L}^*\\| / \\|\\mathcal{L}^*\\|\$", xlabel="iteration number, \$k\$", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), xticks=0:max_n:10000, yticks=yticks, labelfontsize=12, tickfontsize=9)
            savefig(frelplot_alpha, file_dir * pre_dir * "_" * model_name * "_" * method_name * ".pdf")
            timesplot = plot(ts, frel, label=labels, ylabel="\$\\|\\mathcal{L}_k-\\mathcal{L}^*\\| / \\|\\mathcal{L}^*\\|\$", xlabel="time [s]", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), yticks=yticks, labelfontsize=12, tickfontsize=9)
            savefig(timesplot, file_dir * pre_dir * "_t" * "_" * model_name * "_" * method_name * ".pdf")
        end
    end

    return

end

function RUNPaperExperiments_SGL(m, n)
    local model, sol_x, obj, mse, num_nz, t

    file_dir = pwd() * "/figures/"

    methods_list = ["prox-ggnscore", "ssnal", "prox-grad", "bcd"]

    if n <= 2000
        μ = 1.2
    elseif n == 10000
        μ = 1.6
    else
        μ = 2.0
    end

    model_name = "sim_gl"
    grpsize = 100
    use_const_grpsize = true

    result = Dict(method_name=>OrderedDict() for method_name in methods_list)
    labels = String[]
    plt_title = "\$m=$m\$, \$n=$n\$"

    mses = []
    itertimes = []
    mse_ns = []

    for method_name in methods_list
        if method_name in ["prox-ggnscore", "prox-newtonscore", "prox-bfgsscore"]
            α = 1
            λ = 1e-8
        else
            α = nothing
            λ = 1e-7
        end
        @info "Now solving problem $model_name using $method_name..."
        if method_name == "bcd"
            method_label = "BCD"
            γ = λ
            model = GroupLasso(model_name, m, n, grpsize, γ, μ; use_const_grpsize=use_const_grpsize)
            _, obj, mse, itertime, num_nz, t = load_bcd_result(model)
        elseif method_name == "ssnal"
            method_label = "SSNAL"
            γ = λ
            model = GroupLasso(model_name, m, n, grpsize, γ, μ; use_const_grpsize=use_const_grpsize)
            _, obj, mse, itertime, _, num_nz, t = load_ssnal_result(model)
        else
            if n ∈ (5000, 10000) && m == 1000
                x_tol = 1e-6
            else
                x_tol = 1e-9
            end
            model, method_label, sol = solve!(model_name, method_name; m=m, n=n, λ=λ, μ=μ, max_iter=100000, reg_name="gl", grpsize=grpsize, use_const_grpsize=use_const_grpsize, ss_type=1, verbose=2, x_tol=x_tol, α=α)
            obj = sol.obj
            mse = sol.rel
            itertime = sol.times
            sol_x = sol.x
            num_nz = cardcal(sol_x, 0.999)
            t = sol.times[end]
        end

        push!(mses, mse)
        push!(itertimes, max.(itertime,1e-9))
        push!(mse_ns, length(mse))

        push!(labels, method_label)

        result[method_name]["obj"] = sprintf1("%.4E",obj[end])
        result[method_name]["mse"] = sprintf1("%.4E",mse[end])
        result[method_name]["iter"] = sprintf1("%d",length(mse))
        result[method_name]["nnz"] = num_nz
        result[method_name]["time"] = sprintf1("%.2f",t)
    end

    labels = reshape(labels, (1,length(labels)))

    max_n = maximum(mse_ns)
    max_n = Int(round(max_n/min(max_n,10^2), digits=0))

    xticks = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    yticks = [1e2,1e0,1e-2,1e-4,1e-6,1e-8]
    mseplot = plot(mses, label=labels, ylabel="MSE", xlabel="iteration number, \$k\$", xscale=:log10, yscale=:log10, title="$plt_title", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), xticks=xticks, yticks=yticks, labelfontsize=12, tickfontsize=9)
    savefig(mseplot, file_dir * "_$(m)_$(n)_mseplot_" * model_name * ".pdf")

    timeplot = plot(itertimes, mses, label=labels, ylabel="MSE", xlabel="time [s]", yscale=:log10, title="$plt_title", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), yticks=yticks, labelfontsize=12, tickfontsize=9)
    savefig(timeplot, file_dir * "_$(m)_$(n)_timeplot_" * model_name * ".pdf")

    return result, model.x
end

function plot_performance_profile(;n_runs=20)
    local labels, method_label
    file_dir = pwd() * "/figures/"

    problems_list = keys(data_dict)
    methods_list = [Dict("prox-newtonscore"=>0.2), Dict("prox-newtonscore"=>0.5), Dict("prox-newtonscore"=>1.0), "panoc", "zerofpr", "prox-owlqn", "prox-grad", "f-prox-grad"]
    reg_name = "l1"

    n_probs = length(problems_list)

    d = get_density()
    display(d)

    T_all = Dict(name=>OrderedDict() for name in problems_list)

    T = []

    i = 1
    max_iter = 500
    for model_name in problems_list
        labels = String[]
        if model_name == "sim_log"
            λ = 2e-1
            μ = 1.
        else
            λ = 1.
            μ = 1.
        end

        for method_name in methods_list
            if typeof(method_name) == Dict{String, Float64}
                α = [values(method_name)...][1]
                use_method_name = [keys(method_name)...][1]
            else
                α = nothing
                use_method_name = method_name
            end
            @info "Now solving problem $model_name using $use_method_name..."
            m_times = []
            for run_i in 1:n_runs
                model, method_label, sol = solve!(model_name, use_method_name; λ=λ, μ=μ, max_iter=max_iter, reg_name=reg_name, log_reg_problems=problems_list, α=α, x_tol=1e-6, f_tol=1e-6, seed=run_i)
                push!(m_times, sol.times[end])
            end

            if typeof(method_name) == Dict{String, Float64}
                use_this_label = method_label*": \$\\alpha = $α\$"
            elseif method_name in ["prox-newtonscore", "prox-bfgsscore", "prox-ggnscore"]
                use_this_label = method_label*": \$\\alpha = \\max\\{1/L,1\\}\$"
            else
                use_this_label = method_label
            end

            push!(labels, use_this_label)

            m_time = mean(m_times)
            if i <= n_probs
                T_all[model_name][use_this_label] = [m_time]
            else
                push!(T_all[model_name][use_this_label], m_time)
            end
        end
        i += 1
    end
    for label in labels
        method_times = reduce(vcat, [T_all[name][label] for name in problems_list])
        push!(T, method_times)
    end

    perfp = performance_profile(PlotsBackend(), reduce(hcat, T), labels; logscale=false, xlabel="\$\\tau\$", ylabel="\$\\rho(\\tau)\$", linestyle=:auto, legend=:bottomright, _plot_args...)

    savefig(perfp, file_dir * "perf_profile" * ".pdf")

    return
end

# MAIN PLOTS
function plot_allresults(results)

    local data_x, data_y, β, obj_star, x_star

    file_dir = pwd() * "/figures/"

    all_objp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    all_relp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    all_frelp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    all_timesp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    all_msesp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)

    metrics_psnrp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    metrics_msep = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    metrics_rep = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    metrics_splevelp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    deconv_solp = Dict(model_name=>OrderedDict() for model_name in ["deconv-l1", "deconv-l2"])

    for (model, method_name, method_label, sol) in results
        if model.name in ["deconv-l1", "deconv-l2"]
            # keep problem data
            β = model.λ
            data_x = model.x
            data_y = model.y

            metrics_psnrp[model.name][method_label] = sol.metrics["psnr"]
            metrics_msep[model.name][method_label] = sol.metrics["mse"]
            metrics_rep[model.name][method_label] = sol.metrics["re"]
            metrics_splevelp[model.name][method_label] = sol.metrics["sparsity_level"]
            deconv_solp[model.name][method_label] = sol.x

            all_msesp[model.name][method_label] = sol.metrics["mse"][end]
        end

        all_relp[model.name][method_label] = sol.rel
        all_timesp[model.name][method_label] = sol.times
        all_objp[model.name][method_label] = sol.obj
        all_frelp[model.name][method_label] = sol.objrel
    end

    for model_name in keys(all_objp)
        label = [keys(all_objp[model_name])...]
        n = length(label)
        label = reshape(label, (1,n))

        label_all = [keys(all_relp[model_name])...]
        n = length(label_all)
        label_all = reshape(label_all, (1,n))



        if model_name in ["sim_log", "sim_gl"]
            alias_name = "synthetic dataset: \$m=100\$, \$n=4500\$"
        elseif model_name in ["deconv-l1", "deconv-l2"]
            alias_name = ""
        else
            alias_name = model_name * " dataset: \$m=$(data_dict[model_name][1][1])\$, \$n=$(data_dict[model_name][1][2])\$"
        end

        objs = []
        f_rel_ns = []
        for method in keys(all_objp[model_name])
            method_objs = all_objp[model_name][method]
            push!(f_rel_ns, length(method_objs))
            push!(objs, method_objs)
        end
        max_n = maximum(f_rel_ns)
        max_n = Int(round(max_n/min(max_n,5), digits=0))

        yticks = [1e2,1e0,1e-2,1e-4,1e-6,1e-8]

        all_objplot = plot(objs, label=label, ylabel="\$\\mathcal{L}_k\$", xlabel="iteration number, \$k\$", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), xticks=0:max_n:10000, labelfontsize=12, tickfontsize=9)
        savefig(all_objplot, file_dir * "objplot" * "_" * model_name * ".pdf")

        all_frelplot = plot([all_frelp[model_name][method] for method in keys(all_frelp[model_name])], label=label, ylabel="\$\\|\\mathcal{L}_k-\\mathcal{L}^*\\| / \\|\\mathcal{L}^*\\|\$", xlabel="iteration number, \$k\$", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), xticks=0:max_n:10000, yticks=yticks, labelfontsize=12, tickfontsize=9)
        savefig(all_frelplot, file_dir * "frelplot" * "_" * model_name * ".pdf")
        
        all_timesplot = plot([all_timesp[model_name][method] for method in keys(all_timesp[model_name])], [all_relp[model_name][method] for method in keys(all_relp[model_name])], label=label, ylabel="\$\\|x_k-x^*\\|/\\max\\{\\|x^*\\|,1\\}\$", xlabel="time [s]", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf),  yticks=yticks, labelfontsize=12, tickfontsize=9)
        savefig(all_timesplot, file_dir * "timesplot" * "_" * model_name * ".pdf")

        all_relplot = plot([all_relp[model_name][method] for method in keys(all_relp[model_name])], label=label, ylabel="\$\\|x_k-x^*\\|/\\max\\{\\|x^*\\|,1\\}\$", xlabel="iteration number, \$k\$", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), xticks=0:max_n:10000,  yticks=yticks, labelfontsize=12, tickfontsize=9)
        savefig(all_relplot, file_dir * "relplot" * "_" * model_name * ".pdf")


        if model_name in ["deconv-l1", "deconv-l2"]
            label = [keys(deconv_solp[model_name])...]
            n = length(label)
            label = reshape(label, (1,n))
            ts = [all_timesp[model_name][method][end] for method in keys(metrics_psnrp[model_name])]
            iters = [length(all_timesp[model_name][method]) for method in keys(metrics_psnrp[model_name])]
            mses = [all_msesp[model_name][method][end] for method in keys(metrics_psnrp[model_name])]
            names_its_ts_mses = [(label[i], iters[i], mses[i], ts[i]) for i in 1:n]
            title = reshape(vcat(["", ""], [l*", Iter: $i, MSE: $(sprintf1("%.4E",m)), Time [s]: $(round(t, digits=2))" for (l,i,m,t) in names_its_ts_mses]), (1, n+2))

            # plot solutions
            sol_p = [data_x, data_y]
            for method in keys(deconv_solp[model_name])
                push!(sol_p, deconv_solp[model_name][method])
            end
            deconv_solplot = plot(sol_p, layout=(Int((n+2)/2),2), label=["original" "noisy" "" "" "" "" "" ""], title=title, titlefontsize=10, size=(1000, 500), legend=:bottomright)
            savefig(deconv_solplot, file_dir * "sol" * "_" * model_name * ".pdf")

            metrics_psnrplot = plot([metrics_psnrp[model_name][method] for method in keys(metrics_psnrp[model_name])], label=label, ylabel="PSNR [dB]", xlabel="iteration number, \$k\$", size=(450, 250))
            savefig(metrics_psnrplot, file_dir * "psnrplot" * "_" * model_name * ".pdf")

            metrics_mseplot = plot([metrics_msep[model_name][method] for method in keys(metrics_msep[model_name])], label=label, ylabel="mean squared error", xlabel="iteration number, \$k\$", size=(450, 250))
            savefig(metrics_mseplot, file_dir * "mseplot" * "_" * model_name * ".pdf")

            metrics_replot = plot([metrics_rep[model_name][method] for method in keys(metrics_rep[model_name])], label=label, ylabel="absolute error", xlabel="iteration number, \$k\$", size=(450, 250))
            savefig(metrics_replot, file_dir * "REplot" * "_" * model_name * ".pdf")

            metrics_splevelplot = plot([metrics_splevelp[model_name][method] for method in keys(metrics_splevelp[model_name])], label=label, ylabel="sparsity level", xlabel="iteration number, \$k\$", size=(450, 250))
            savefig(metrics_splevelplot, file_dir * "splevelplot" * "_" * model_name * ".pdf")
        end

    end

end


