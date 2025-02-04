using MAT

function load_ssnal_result(model)
    file_dir = pwd() * "/ssnal/"
    m, n = size(model.A)
    file = matopen(file_dir*"ssnal_sol_$(m)_$(n).mat")
    sol = vec(read(file, "x"))
    obj = read(file, "primobj")
    mse = vec(read(file, "mses"))
    itertime = vec(read(file, "itertimes"))
    t = read(file, "ttime")
    num_nz0 = Int(read(file, "num_nz"))
    close(file)
    num_nz = nnz(sparse(sol))
    return sol, obj, mse, itertime, num_nz, num_nz0, t
end

function cardcal(x, r)
    local k
    n = length(x)
    normx1 = norm(x, 1)
    absx = sort(abs.(x), rev=true)
    for i = 1:n
        if sum(absx[1:i]) >= r * normx1
            k = i
            break
        end
    end
    return k
end