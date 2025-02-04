using NPZ

include("load_ssnal.jl")

function load_bcd_result(model)
    file_dir = pwd() * "/bcd/"
    m, n = size(model.A)
    sol = vec(npzread(file_dir*"$(m)_$(n)_bcd_sol.npy"))
    obj = model.f(sol) + get_reg(model, sol, "gl")
    mse = npzread(file_dir*"$(m)_$(n)_mses.npy")
    itertime = npzread(file_dir*"$(m)_$(n)_iter_times.npy")
    t = npzread(file_dir*"times_sgl_$(m)_$(n).npy")
    num_nz = cardcal(sol, 0.999)
    return sol, obj, mse, itertime, num_nz, t[1]
end