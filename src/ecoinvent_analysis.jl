## installs
import Pkg
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Statistics")
Pkg.add("MultivariateStats")
Pkg.add("Random")
Pkg.add("Plots")
Pkg.add("PlotlyJS")
Pkg.add("MLJBase")
Pkg.add("MLJModels")
Pkg.add("ChemometricsTools")
Pkg.add("PartialLeastSquaresRegressor")
Pkg.add("GLM")

## load dataset
using DataFrames
using CSV
using Statistics
raw_data = read("./data/lcia371.csv", String);
io = IOBuffer(raw_data);
df = CSV.File(io,
    header=1:2,
    delim=',',
    ignoreemptylines=true,
    decimal='.',
) |> DataFrame;

## select data
selectdataset(a) = select!(a, r"name", r"ReCiPe Midpoint \(E\) V1.13:")
removedatasetname(a) = rename!(n -> n[1:4] == "name" ? "name" : join(deleteat!(split(n, ":"), 1)), a)
modifyunit(a) = rename!(n -> n[1:4] == "name" ? "name" : join([split(n, "_")[1], " [", split(n, "_")[2], "]"]), a)
modifyname(a) = a |> removedatasetname |> modifyunit

recipe_df = copy(df)

recipe_df |> selectdataset |> modifyname |> dropmissing!

## PCA
using MultivariateStats

# remove name column and perform pca
recipe_data = Matrix(select(recipe_df, Not([:name,])))
pca = fit(PCA, recipe_data)

## reduce dataset
filterbrick(a) = filter(a -> occursin("brick", a.name), a)
filterconcrete(a) = filter(a -> occursin("concrete block", a.name), a)
filtermarketfor(a) = filter(a -> occursin("market for", a.name), a)
filterproduction(a) = filter(a -> !occursin("production", a.name), a)

getbricks = a -> a |> filterbrick |> filtermarketfor |> filterproduction
getconcrete = a -> a |> filterconcrete |> filtermarketfor |> filterproduction

bricks_df = getbricks(recipe_df)
concrete_df = getconcrete(recipe_df)

getconcretelabel = function(a)
    if (occursin("aerated", a))
        return "aerated"
    elseif (occursin("lightweight", a))
        return "lightweight"
    else
        return "regular"
    end
end

concrete_df.label = map(a -> getconcretelabel(a), concrete_df.name)

brickconcrete_df = vcat(bricks_df, concrete_df)

## classify concrete/brick
using Random

X = Matrix(brickconcrete_df)

# shuffle matrix
X = X[shuffle(1:end), :]

# scaling
labels = X[:, 1]
X = X[:, 2:end]
X = (X .- mean(X, dims = 1))./ std(X, dims = 1)

removenthrow = (M, n, o = 0) -> M[filter(a -> !iszero((a-o)%n), 1:end), :]

# training set
Xtr = convert(Array{Float64}, removenthrow(X, 5, 1))'
Xtr_labels = vec(convert(Array, removenthrow(labels, 5, 1)))

# testing set
Xte = convert(Array{Float64}, X[1:5:end, :])'
Xte_labels = convert(Array, labels[1:5:end])

# train PCA model
M = fit(PCA, Xtr)

# scree plot
using Plots
plotlyjs()
p = plot(principalvars(M), label = "Principal component variance", marker = :circle, linewidth = 1)
hline!([1], label = "y = 1")
savefig(p, "img/scree.png")

# apply PCA model to testing set
Yte = MultivariateStats.transform(M, Xte)
Ytr = MultivariateStats.transform(M, Xtr)

# reconstruct testing observations (approximately)
Xr = reconstruct(M, Yte)

# group results by testing set labels for color coding
brick_te = Yte[:, occursin.("brick", Xte_labels)]
brick_tr = Ytr[:, occursin.("brick", Xtr_labels)]
brick_labels_te = Xte_labels[occursin.("brick", Xte_labels)]
brick_labels_tr = Xtr_labels[occursin.("brick", Xtr_labels)]
concrete_te = Yte[:, occursin.("concrete", Xte_labels)]
concrete_tr = Ytr[:, occursin.("concrete", Xtr_labels)]
concrete_labels_te = Xte_labels[occursin.("concrete", Xte_labels)]
concrete_labels_tr = Xtr_labels[occursin.("concrete", Xtr_labels)]

# visualise principal components
for i in 1:3
    v1 = i; v2 = i%3+1;
    p = plot(scatter(brick_te[v1,:], brick_te[v2,:], marker = :square, label = "Brick (testing)", color = :blue, linewidth = 0, hover=brick_labels_te))
    scatter!(brick_tr[v1,:], brick_tr[v2,:], marker = :circle, label = "Brick (training)", color = :blue, linewidth = 0, hover=brick_labels_tr)
    scatter!(concrete_te[v1,:], concrete_te[v2,:], marker = :square, label = "Concrete (testing)", color = :orange, linewidth = 0, hover=concrete_labels_te)
    scatter!(concrete_tr[v1,:], concrete_tr[v2,:], marker = :circle, label = "Concrete (training)", color = :orange,linewidth = 0, hover=concrete_labels_tr)
    plot!(p, xlabel=string("PC", v1), ylabel=string("PC", v2))
    gui()
    savefig(p, string("img/pc", v1, "pc", v2, ".png"))
end

## PLS fitting
using MLJBase, MLJModels, GLM, Plots.PlotMeasures
@load PLSRegressor pkg=PartialLeastSquaresRegressor

recipe_endpoint_df = select(df, r"name", r"ReCiPe Endpoint \(I,A\):total:total")

recipe_endpoint_df |> modifyname |> dropmissing!
concrete_endpoint_df = getconcrete(recipe_endpoint_df)

endpoint_df = hcat(concrete_endpoint_df, concrete_df[:, 2:end])
data = endpoint_df[:, 2:end]
rename!(data, 1 => :total)

y, X = unpack(data, ==(:total), colname -> true)
varnames = [replace(x, r"[^a-z|^\s].*" => "") for x in names(X)]
rename!(X, varnames)

regressor = PartialLeastSquaresRegressor.PLSRegressor(n_factors = 3)
pls_model = @pipeline Standardizer regressor target=Standardizer

train, test = partition(eachindex(y), 0.7, shuffle=true)

pls_machine = machine(pls_model, X, y)

fit!(pls_machine, rows = train)

fp = fitted_params(pls_machine)
weights = fp.pls_regressor.fitresult.W

gr()
p = plot(xlabel="Variable", ylabel="Relative weight", legend=:outertopleft, width=900px, height=700px)
hline!([0], label=false, color=:black)
plot!(1:18, weights[:,1], xticks=(1:18, varnames), xrotation=30, marker=:circle, label="Weights LV 1", xlims=(0,20))
plot!(1:18, weights[:,2], marker=:circle, label="Weights LV 2")
savefig(p, "img/pls_weights.png")

cv_ŷ = MLJBase.predict(pls_machine, rows = test)
cal_ŷ = MLJBase.predict(pls_machine, rows = train)

cv_mean = rmse(cv_ŷ, y[test]) |> mean
cal_mean = rmse(cal_ŷ, y[train]) |> mean

cv_df = DataFrame(real = y[test], pred = cv_ŷ)
cal_df = DataFrame(real = y[train], pred = cal_ŷ)

cv_ols = lm(@formula(pred ~ real), cv_df)
r2(cv_ols)

cal_ols = lm(@formula(pred ~ real), cal_df)
r2(cal_ols)
cal_model(x) = coef(cal_ols)[1] + coef(cal_ols)[2]*x

p = plot([0,0.15], [0, 0.15], linewidth=1, label = "x = y", color=:black)
scatter!(y[test], cv_ŷ, label = "Cross-validation")
scatter!(y[train], cal_ŷ, label = "Calibration")
plot!(y[train], cal_model.(y[train]), label = "Prediction fit", color=:green)
plot!(p, xlabel = "Actual value", ylabel="Predicted value")
savefig(p, "img/pls.png")

#=
## PLS using ChemometricsTools
using ChemometricsTools
recipe_endpoint_df = select(df, r"name", r"ReCiPe Endpoint \(I,A\):total:total")

recipe_endpoint_df |> modifyname |> dropmissing!
concrete_endpoint_df = getconcrete(recipe_endpoint_df)

endpoint_df = hcat(concrete_endpoint_df, concrete_df[:, 2:end])
data = endpoint_df[:, 2:end]
rename!(data, 1 => :total)

y, X = unpack(data, ==(:total), colname -> true)

((X_cal, y_cal), (X_cv, y_cv)) = SplitByProportion(Matrix(X), y, 0.7)
msc_obj = MultiplicativeScatterCorrection(X_cal)
X_cal = msc_obj(X_cal)
X_cv = msc_obj(X_cv)

lv = 8
err = repeat([0.0], lv)

for l in 1:lv, (Fold, HoldOut) in KFoldsValidation(20, X_cal, y_cal)
    plsr = PartialLeastSquares(Fold[1], Fold[2]; Factors = l)
    err[l] += SSE(plsr(HoldOut[1]), HoldOut[2])
end
scatter(err, xlabel = "Latent variables", ylabel = "Cumulative SSE", labels = ["Error"])
bestlv = argmin(err)
plsr = PartialLeastSquares(X_cal, y_cal; Factors = bestlv)
RMSE(plsr(X_cv), y_cv)
hot = Hotelling(X_cal, plsr)

p = plot([0,0.15], [0, 0.15], linewidth=1, label = "x = y", color=:black)
scatter!(y[test], cv_ŷ, label = "Cross-validation")
scatter!(y[train], cal_ŷ, label = "Calibration")
plot!(y[train], cal_model.(y[train]), label = "Prediction fit", color=:green)
plot!(p, xlabel = "Actual value", ylabel="Predicted value")
savefig(p, "img/pls.png")

=#




