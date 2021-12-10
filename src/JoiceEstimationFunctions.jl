#=
JoiceEstimationFunctions:
- Julia version: 1.6
- Author: connorforsythe
- Date: 2021-11-07
=#


using JuMP, NLopt, LinearAlgebra, Statistics, PyCall, ProgressMeter, Distributions, FiniteDiff

export EstimateModel, testFun

function estimateHessianUsingObjective(modelInfo::ModelInfo, theta::T...) where {T<:Real}

    logitVals::Union{Vector{T}, Matrix{T}} = logit(modelInfo.Data, modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, modelInfo.ChoiceSetIDColumn, modelInfo.ObservationIDColumn, modelInfo.ModelSpace, modelInfo.DataStructure, theta...)


    if typeof(logitVals)==Matrix{T}

        parameterDraws::Union{Matrix{T}, Nothing} = getSimulatedParameters(modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, theta...)
        normalDraws::Union{Matrix{T}, Nothing} = getStandardNormalDraws(modelInfo.RandomDraws)
        df::Union{DataFrame, Nothing} = DataFrame(logitVals, :auto)

    else
        parameterDraws = nothing
        normalDraws = nothing
        df = nothing
    end

    obj(x::Vector{T}) = getObjLM(logitVals, df, parameterDraws, normalDraws, modelInfo.Data, modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, modelInfo.ChoiceSetIDColumn, modelInfo.ObservationIDColumn, modelInfo.ChoiceColumn, modelInfo.ModelSpace, modelInfo.DataStructure, x...)

    hess = FiniteDiff.finite_difference_hessian(obj, [theta...])

    return hess
end

function getObjLM(logitVals::AbstractMatrix{T}, df::DataFrame, parameterDraws::AbstractMatrix{T}, normalDraws::AbstractMatrix{T}, data::DataFrame, parameters::Vector{Symbol}, randomParameters::Dict{Symbol, String}, randomDraws::Matrix{T}, choiceSetIDColumn::Symbol, observationIDColumn::Symbol, choiceColumn::Symbol, modelSpace::Preference, dataStructure::Panel, theta::T...) where {T<:Real}
    logit!(logitVals, parameterDraws, normalDraws, data, parameters, randomParameters, randomDraws, choiceSetIDColumn, observationIDColumn, modelSpace, dataStructure, theta...)
    return logLikelihoodLM(logitVals, data[:, choiceColumn], data[:, observationIDColumn], df)
end

function getObjLM(logitVals::AbstractVector{T}, df::Nothing, parameterDraws::Nothing, normalDraws::Nothing, data::DataFrame, parameters::Vector{Symbol}, randomParameters::Nothing, randomDraws::Nothing, choiceSetIDColumn::Symbol, observationIDColumn::Symbol, choiceColumn::Symbol, modelSpace::Preference, dataStructure::Panel, theta::T...) where {T<:Real}
    logit!(logitVals, parameterDraws, normalDraws, data, parameters, randomParameters, randomDraws, choiceSetIDColumn, observationIDColumn, modelSpace, dataStructure, theta...)
    return logLikelihoodLM(logitVals, data[:, choiceColumn], data[:, observationIDColumn], df)
end
function EstimateModel(modelInfo::ModelInfo; numMultistarts::Int = 1, post::Bool = true)

    startingVals::Vector{T} where {T<:Real} = getStartingValues(modelInfo)
    res = EstimateModel(modelInfo, startingVals...; numMultistarts=numMultistarts, post=post)

    return res

end

function EstimateModel(modelInfo::ModelInfo, theta0::T...; numMultistarts::Int = 1, post::Bool = true) where {T<:Real}
    println("Beginning estimation procedure...")
    res = buildAndEstimateJuMPModel(modelInfo, theta0...; numMultistarts=numMultistarts, post=post)
    println("Estimation procedure completed!")
    return res

end

function EstimateModelLM(modelInfo::ModelInfo; numMultistarts::Int = 1, post::Bool = true)

    startingVals::Vector{T} where {T<:Real} = getStartingValues(modelInfo)
    println("Starting values: $startingVals")
    res = EstimateModelLM(modelInfo, startingVals...; numMultistarts=numMultistarts, post=post)

    return res

end

function EstimateModelLM(modelInfo::ModelInfo, theta0::T...; numMultistarts::Int = 1, post::Bool = true) where {T<:Real}
    @time logitVals::Union{Vector{T}, Matrix{T}} = logit(modelInfo.Data, modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, modelInfo.ChoiceSetIDColumn, modelInfo.ObservationIDColumn, modelInfo.ModelSpace, modelInfo.DataStructure, theta0...)

    logitVals = logitVals.*0

    if typeof(logitVals)==Matrix{T}
        println("This is true")
        logitVals = Matrix{T}(undef, ncol(modelInfo.Data), modelInfo.InterDrawCount)
        parameterDraws::Matrix{T} = getSimulatedParameters(modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, theta0...)
        normalDraws::Matrix{T} = getStandardNormalDraws(modelInfo.RandomDraws)

        @time logit!(logitVals, parameterDraws, normalDraws, modelInfo.Data, modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, modelInfo.ChoiceSetIDColumn, modelInfo.ObservationIDColumn, modelInfo.ModelSpace, modelInfo.DataStructure, theta0...)
    else
        @time logit!(logitVals, modelInfo.Data, modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, modelInfo.ChoiceSetIDColumn, modelInfo.ObservationIDColumn, modelInfo.ModelSpace, modelInfo.DataStructure, theta0...)
    end



    @time logit!(logitVals, modelInfo.Data, modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, modelInfo.ChoiceSetIDColumn, modelInfo.ObservationIDColumn, modelInfo.ModelSpace, modelInfo.DataStructure, theta0...)
    println("-------")

end

function createMemoObjGrad(objGradFun::Function)
    prevX, prevObjGrad = nothing, nothing
    function memoGetObjGrad(i, x::T...) where{T<:Real}
        if(x!=prevX)
            prevX, prevObjGrad = x, objGradFun(x...)
        end
        return prevObjGrad[i]
    end

    return [(x...)-> memoGetObjGrad(i, x...) for i in 1:2]

end

function buildAndEstimateJuMPModel(modelInfo::ModelInfo, theta0::T...; numMultistarts::Int = 1, post::Bool=true) where{T<:Real}
    println("Constructing JuMP Model...")
    objGradToMemo(x...) = getObjGrad(modelInfo, x...)

    memoObjGrad = createMemoObjGrad(objGradToMemo)

    numMultistarts = 1

    numParams::Int = length(theta0)


    obj(x...) = memoObjGrad[1](x...)
    grad(x...) = memoObjGrad[2](x...)


    scaledGrad(x...) = memoObjGrad[2](x...)./10
    jumpGrad(g::AbstractVector{T}, x::T...) where{T<:Real} = placeGradInVector(g, scaledGrad, x...)

    rModel = Model(NLopt.Optimizer)
    set_optimizer_attribute(rModel, "algorithm", :LD_LBFGS)
    set_optimizer_attribute(rModel, "xtol_rel", 1e-4)
    set_optimizer_attribute(rModel, "ftol_rel", 1e-4)
    set_optimizer_attribute(rModel, "xtol_abs", 1e-4)
    set_optimizer_attribute(rModel, "ftol_abs", 1e-4)
    set_optimizer_attribute(rModel, "maxeval", 1000)
    register(rModel, :LL, numParams, obj, jumpGrad; autodiff=false)
    @variable(rModel, theta[1:numParams])

    for i in 1:length(theta0)

        JuMP.set_start_value(theta[i], theta0[i])

    end

    @NLobjective(rModel, Max, LL(theta...))
    println("JuMP Model Constructed...")
    println("JuMP Model Initial Estimation...")
    maxLL = -Inf
    bestParams = [theta0...]
    try
        JuMP.optimize!(rModel)

        maxLL = objective_value(rModel)
        bestParams = value.(theta)
    catch e
    end
    println("JuMP Model Initial Estimation Complete...")

    if(numMultistarts>1)
        randomDraws = getDraws(numParams, numMultistarts-1)
        parameterDraws = min(modelInfo.PreferenceSearchBounds...).+randomDraws.*(max(modelInfo.PreferenceSearchBounds...)-min(modelInfo.PreferenceSearchBounds...))

        pMeter = Progress(numMultistarts-1, desc="Running Multistarts:")
        for j in 1:(numMultistarts-1), i in 1:numParams
            JuMP.set_start_value(theta[i], parameterDraws[i, j])
            if(i==numParams)
                try
                    JuMP.optimize!(rModel)

                    maxLL = objective_value(rModel)
                    bestParams = value.(theta)

                    if(objective_value(rModel)>maxLL)
                        maxLL = objective_value(rModel)
                        bestParams = value.(theta)
                    end
                catch e
                end
                next!(pMeter)
            end
        end

    end

    println("Refining Model Estimates...")

    set_optimizer_attribute(rModel, "xtol_rel", 1e-15)
    set_optimizer_attribute(rModel, "ftol_rel", 1e-15)
    set_optimizer_attribute(rModel, "xtol_abs", 1e-15)
    set_optimizer_attribute(rModel, "ftol_abs", 1e-15)
    set_optimizer_attribute(rModel, "maxeval", 10000)

    for i in 1:length(bestParams)
        JuMP.set_start_value(theta[i], bestParams[i])
    end



    try
        JuMP.optimize!(rModel)

        maxLL = objective_value(rModel)
        bestParams = value.(theta)
    catch e
        println("Issue with refining estimates - try a different starting point.")
        println("Returning nothing.")
        return nothing
    end
    println("Model Estimates Refined...")
    println("Post-Estimation Calculations Beginning...")
    r = Dict{Any, Any}()
    coefNames::Vector{Any} = getParameterNames(modelInfo.Parameters, modelInfo.RandomParameters)
    r["LL"] = maxLL
    if post
        r["coef"] = NamedTuple{Tuple(Symbol.(coefNames))}(Tuple(bestParams))
    else
        r["coef"] = [bestParams...]
    end

    r["grad"] = grad(bestParams...)

    if post
        obsCount::Int = length(unique(modelInfo.Data[:, modelInfo.ObservationIDColumn]))

        r["hessian"] = estimateHessianUsingObjective(modelInfo, bestParams...)

        r["cov"] = -inv(r["hessian"])
        r["se"] = NamedTuple{Tuple(Symbol.(coefNames))}(Tuple(sqrt.(diag(r["cov"]))))

        r["obsGrad"] = getObservationGradients(modelInfo, value.(theta)...)

        meanGradMat = repeat(mean(r["obsGrad"], dims=1), size(r["obsGrad"])[1])

        r["robustCov"] = r["cov"]*(transpose(r["obsGrad"].-meanGradMat)*(r["obsGrad"].-meanGradMat))*r["cov"]
        r["robustCov"] = (obsCount/(obsCount-1)).*r["robustCov"]
        r["robustSE"] = NamedTuple{Tuple(Symbol.(coefNames))}(Tuple(sqrt.(diag(r["robustCov"]))))
        println("Post-Estimation Calculations Completed...")
    end
    println("Model Estimated!")
    return r

end

function getParameterNames(parameters::Vector{Symbol}, randomParameters::Nothing) return string.(parameters) end
function getParameterNames(parameters::Vector{Symbol}, randomParameters::Dict{Symbol, String})
    r = string.(deepcopy(parameters))
    for coef in keys(randomParameters)
        push!(r, string(coef, "-μ"))
    end

    for coef in keys(randomParameters)
        push!(r, string(coef, "-σ"))
    end

    return r
end

function getHessianNumericallyFromObjective(obj::Function, x::T...) where {T<:Real}
    passFun(x::Vector{T}) = obj(x...)
    hess = FiniteDiff.finite_difference_hessian(passFun, [x...])

    return hess

end

function placeGradInVector(g::AbstractVector{T}, fun::Function, x::T...) where{T}
    r = fun(x...)
    for i in 1:length(x)
        g[i] = r[i]
    end
end

function getObjGrad(modelInfo::ModelInfo, theta::T...) where {T<:Real}

    logitVals::Union{Vector{T}, Matrix{T}} = logit(modelInfo.Data, modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, modelInfo.ChoiceSetIDColumn, modelInfo.ObservationIDColumn, modelInfo.ModelSpace, modelInfo.DataStructure, theta...)
    parameterCols = getParameterColumns(modelInfo.Parameters, modelInfo.RandomParameters)
    logLikelihoodVal::T = logLikelihood(logitVals, modelInfo.Data[:, modelInfo.ChoiceColumn], modelInfo.Data[:, modelInfo.ObservationIDColumn])
    logLikelihoodGradVals::Vector{T} = logLikelihoodGrad(logitVals, modelInfo.Data[:, modelInfo.ChoiceColumn], Matrix(modelInfo.Data[:, parameterCols]), modelInfo.Data[:, modelInfo.ObservationIDColumn], modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, theta...)


    return logLikelihoodVal, logLikelihoodGradVals
end

function getObjGradLM(logitVals::AbstractVecOrMat{T}, modelInfo::ModelInfo, theta::T...) where {T<:Real}

    logit!(logitVals, modelInfo.Data, modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, modelInfo.ChoiceSetIDColumn, modelInfo.ObservationIDColumn, modelInfo.ModelSpace, modelInfo.DataStructure, theta...)
    parameterCols = getParameterColumns(modelInfo.Parameters, modelInfo.RandomParameters)
    logLikelihoodVal::T = logLikelihood(logitVals, modelInfo.Data[:, modelInfo.ChoiceColumn], modelInfo.Data[:, modelInfo.ObservationIDColumn])
    logLikelihoodGradVals::Vector{T} = logLikelihoodGrad(logitVals, modelInfo.Data[:, modelInfo.ChoiceColumn], Matrix(modelInfo.Data[:, parameterCols]), modelInfo.Data[:, modelInfo.ObservationIDColumn], modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, theta...)


    return logLikelihoodVal, logLikelihoodGradVals
end


function getParameterColumns(parameters::Vector{Symbol}, randomParameters::Nothing)

    return parameters

end

function getParameterColumns(parameters::Vector{Symbol}, randomParameters::Dict{Symbol, String})
    r = deepcopy(parameters)

    push!(r, keys(randomParameters)...)

    return r

end

function getObservationGradients(modelInfo::ModelInfo, theta::T...) where {T<:Real}

    obsCol = modelInfo.ObservationIDColumn

    logitVals::Union{Vector{T}, Matrix{T}} = logit(modelInfo.Data, modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, modelInfo.ChoiceSetIDColumn, modelInfo.ObservationIDColumn, modelInfo.ModelSpace, modelInfo.DataStructure, theta...)
    uniqueObsIDs = unique(modelInfo.Data[:, obsCol])

    obsGrads = Matrix{T}(undef, length(uniqueObsIDs), length(theta))

    i::Int = 1

    pMeter = Progress(length(uniqueObsIDs); desc="Computing observation gradients for covariance estimation: " )

    for tempID in uniqueObsIDs

        tempObsIndices = modelInfo.Data[:, obsCol].==repeat([tempID], nrow(modelInfo.Data))
        tempObs = modelInfo.Data[tempObsIndices, obsCol]
        parameterCols = getParameterColumns(modelInfo.Parameters, modelInfo.RandomParameters)
        tempXMat = Matrix(modelInfo.Data[tempObsIndices, parameterCols])
        tempChoices = modelInfo.Data[tempObsIndices, modelInfo.ChoiceColumn]
        if(modelInfo.RandomParameters!=nothing)
            tempLogit = logitVals[tempObsIndices, :]
        else
            tempLogit = logitVals[tempObsIndices]
        end

        tempGrad::Vector{T} = logLikelihoodGrad(tempLogit, tempChoices, tempXMat, tempObs, modelInfo.Parameters, modelInfo.RandomParameters, modelInfo.RandomDraws, theta...)


        obsGrads[i, 1:length(theta)] = tempGrad

        i+=1
        next!(pMeter)



    end

    return obsGrads

end


function logit(data::DataFrame, parameters::Vector{Symbol}, randomParameters::Nothing, randomDraws::Nothing, choiceSetIDColumn::Symbol, observationIDColumn::Symbol, modelSpace::Preference, dataStructure::Union{Cross, Panel}, theta::T...) where {T<:Real}

    tempDF::DataFrame = DataFrame(expU = exp.(Matrix(data[:, parameters])*[theta...]))
    tempDF[:, choiceSetIDColumn]=data[:, choiceSetIDColumn]

    gData::GroupedDataFrame = groupby(tempDF, choiceSetIDColumn)
    gDataRes::DataFrame = combine(gData, :expU => sum)

    tempDF = innerjoin(tempDF, gDataRes, on=choiceSetIDColumn) #TODO fix so that obsID is forced to be unique

    tempDF[:, :p] .= tempDF[:, :expU]./tempDF[:, :expU_sum]
    r::Vector{T} = tempDF[:, :p]

    return r
end

function logit!(logitVals::AbstractMatrix{T}, index::Int, data::DataFrame, parameters::Vector{Symbol}, randomParameters::Nothing, randomDraws::Nothing, choiceSetIDColumn::Symbol, observationIDColumn::Symbol, modelSpace::Preference, dataStructure::Union{Cross, Panel}, theta::T...) where {T<:Real}

    tempDF::DataFrame = DataFrame(expU = exp.(Matrix(data[:, parameters])*[theta...]))
    tempDF[:, choiceSetIDColumn]=data[:, choiceSetIDColumn]

    gData::GroupedDataFrame = groupby(tempDF, choiceSetIDColumn)
    gDataRes::DataFrame = combine(gData, :expU => sum)

    tempDF = innerjoin(tempDF, gDataRes, on=choiceSetIDColumn) #TODO fix so that obsID is forced to be unique

    tempDF[:, :p] .= tempDF[:, :expU]./tempDF[:, :expU_sum]
    logitVals[:, index] .= tempDF[:, :p]

end

function logit!(logitVals::AbstractVector{T}, parameterDraws::Nothing, normalDraws::Nothing, data::DataFrame, parameters::Vector{Symbol}, randomParameters::Nothing, randomDraws::Nothing, choiceSetIDColumn::Symbol, observationIDColumn::Symbol, modelSpace::Preference, dataStructure::Union{Cross, Panel}, theta::T...) where {T<:Real}

    tempDF::DataFrame = DataFrame(expU = exp.(Matrix(data[:, parameters])*[theta...]))
    tempDF[:, choiceSetIDColumn]=data[:, choiceSetIDColumn]

    gData::GroupedDataFrame = groupby(tempDF, choiceSetIDColumn)
    gDataRes::DataFrame = combine(gData, :expU => sum)

    tempDF = innerjoin(tempDF, gDataRes, on=choiceSetIDColumn) #TODO fix so that obsID is forced to be unique

    tempDF[:, :p] .= tempDF[:, :expU]./tempDF[:, :expU_sum]
    logitVals .= tempDF[:, :p]

end

function logit(data::DataFrame, parameters::Vector{Symbol}, randomParameters::Dict{Symbol, String}, randomDraws::Matrix{T}, choiceSetIDColumn::Symbol, observationIDColumn::Symbol, modelSpace::Preference, dataStructure::Panel, theta::T...) where {T<:Real}

    parameterDraws::Matrix{T} = getSimulatedParameters(parameters, randomParameters, randomDraws, theta...)
    normalDraws::Matrix{T} = getStandardNormalDraws(randomDraws)


    paramsInOrder = [parameters..., keys(randomParameters)...]

    numDraws = size(normalDraws)[2]

    logitVals = Matrix{T}(undef, size(data)[1], numDraws)

    for i in 1:numDraws
        logitVals[:, i] = logit(data, paramsInOrder, nothing, nothing, choiceSetIDColumn, observationIDColumn, modelSpace, dataStructure, parameterDraws[:, i]...)
    end

    return logitVals

end

function logit!(logitVals::AbstractMatrix{T}, parameterDraws::AbstractMatrix{T}, normalDraws::AbstractMatrix{T}, data::DataFrame, parameters::Vector{Symbol}, randomParameters::Dict{Symbol, String}, randomDraws::Matrix{T}, choiceSetIDColumn::Symbol, observationIDColumn::Symbol, modelSpace::Preference, dataStructure::Panel, theta::T...) where {T<:Real}

    parameterDraws = getSimulatedParameters(parameters, randomParameters, randomDraws, theta...)
    normalDraws = getStandardNormalDraws(randomDraws)


    paramsInOrder = [parameters..., keys(randomParameters)...]

    numDraws = size(normalDraws)[2]

    for i in 1:numDraws
        logit!(logitVals, i, data, paramsInOrder, nothing, nothing, choiceSetIDColumn, observationIDColumn, modelSpace, dataStructure, parameterDraws[:, i]...)
    end

end

function logLikelihood(logit::Vector{L}, choices::Vector{C}, observationIDs::Vector{O}) where {L<:Real, C<:Real, O<:Any}

    return sum(log.(logit).*choices)

end

function logLikelihood(logit::Matrix{L}, choices::Vector{C}, observationIDs::Vector{O}) where {L<:Real, C<:Real, O<:Any}
    logitDF::DataFrame = getIndividualLogits(logit, choices, observationIDs)
    meanLogit::Matrix{L} = mean(Matrix(logitDF[:, 2:size(logitDF)[2]]), dims=2)

    return sum(log.(meanLogit))

end

function logLikelihoodLM(logit::Vector{L}, choices::Vector{C}, observationIDs::Vector{O}, df::Nothing) where {L<:Real, C<:Real, O<:Any}

    return logLikelihood(logit, choices, observationIDs)

end

function logLikelihoodLM(logit::Matrix{L}, choices::Vector{C}, observationIDs::Vector{O}, df::DataFrame) where {L<:Real, C<:Real, O<:Any}

    meanLogit::Matrix{L} = mean(Matrix(getIndividualLogitsLM(logit, choices, observationIDs, df)[:, 2:size(df)[2]-1]), dims=2)

    return sum(log.(meanLogit))

end


function getIndividualLogits(logit::Matrix{L}, choices::Vector{C}, observationIDs::Vector{O}) where {L<:Real, C<:Real, O<:Any}

    logitCopy = deepcopy(logit)

    for i in 1:size(logitCopy)[2]
        logitCopy[:, i] = logitCopy[:, i].^choices
    end



    df::DataFrame = DataFrame(logitCopy, :auto)
    obsNames = Symbol.([names(df)...])
    df[:, :obs] .= observationIDs

    gData::GroupedDataFrame = groupby(df, :obs)
    df = combine(gData, obsNames .=> prod)

    return df

end

function getIndividualLogitsLM(logit::Matrix{L}, choices::Vector{C}, observationIDs::Vector{O}, df::DataFrame) where {L<:Real, C<:Real, O<:Any}

    for i in 1:size(logit)[2]
        logit[:, i] = logit[:, i].^choices
        df[:, Symbol(string("x", i))] .= logit[:, i]
    end

    obsNames = Symbol.([names(df)...])

    df[:, :obs] .= observationIDs

    gData::GroupedDataFrame = groupby(df, :obs)

    return combine(gData, obsNames[2:end] .=> prod)

end

function logLikelihoodGrad(logit::Vector{L}, choices::Vector{C}, xMat::Matrix{M}, observationIDs::Vector{O}, parameters::Vector{Symbol}, randomParameters::Nothing, randomDraws::Nothing, theta::L...) where{L<:Real, C<:Real, M<:Real, O<:Any}

    res::Vector{L} = choices.-logit

    gradient::Vector{L} = transpose(xMat)*res

    return gradient

end

function logLikelihoodGrad(logit::Matrix{L}, choices::Vector{C}, xMat::Matrix{M}, observationIDs::Vector{O}, parameters::Vector{Symbol}, randomParameters::Dict{Symbol, String}, randomDraws::Matrix{L}, theta::L...) where{L<:Real, C<:Real, M<:Real, O<:Any}

    res::Matrix{L} = Matrix{L}(undef, size(logit)...)

    for i in 1:size(logit)[2]
        res[:, i] = choices.-logit[:, i]
    end

    fixedInds = 1:length(parameters)
    randomInds = (length(parameters)+1):size(xMat)[2]

    individualLogits::DataFrame = getIndividualLogits(logit, choices, observationIDs)
    meanLogit::Matrix{L} = mean(Matrix(individualLogits[:, 2:size(individualLogits)[2]]), dims=2)
    ll::L = sum(log.(meanLogit))

    normalDraws::Matrix{L} = getStandardNormalDraws(randomDraws)
    grad = repeat([0.], length(theta))

    #Perform calculations once in order to pre-alloc memory for temporary variables in loops below
    #outer loop
    tempObs::O = individualLogits[1,:obs]
    tempInds::BitVector = observationIDs.==tempObs
    tempFixedXMat::Matrix{L} = xMat[tempInds, :]
    indGrad = repeat([0.], length(theta))
    meanIndLogit::L = 0.

    #inner loop
    i::Int = 1
    tempRes::Vector{L} = res[tempInds, i]
    tempLogit::L = individualLogits[1, Symbol(string("x", i, "_prod"))]
    tempNormalDrawMat::Matrix{L} = transpose(reshape(repeat(normalDraws[:, i], size(tempFixedXMat)[1]), length(randomParameters), size(tempFixedXMat)[1]))
    partialMat::Matrix{L} = hcat(tempFixedXMat, tempFixedXMat[:, randomInds].*tempNormalDrawMat)
    for tempRow in eachrow(individualLogits)
        tempObs = tempRow[:obs]
        tempInds = observationIDs.==tempObs
        tempFixedXMat = xMat[tempInds, :]
        indGrad = repeat([0.], length(theta))
        meanIndLogit = 0.
        for i in 1:size(logit)[2]
            tempRes = res[tempInds, i]
            tempLogit = tempRow[Symbol(string("x", i, "_prod"))]
            meanIndLogit+=tempLogit
            tempNormalDrawMat= transpose(reshape(repeat(normalDraws[:, i], size(tempFixedXMat)[1]), length(randomParameters), size(tempFixedXMat)[1]))

            partialMat = hcat(tempFixedXMat, tempFixedXMat[:, randomInds].*tempNormalDrawMat)

            indGrad .= indGrad.+(tempLogit.*(transpose(partialMat)*tempRes))

        end
        grad = grad.+(indGrad./(meanIndLogit/size(logit)[2]))
    end

    grad = grad./(size(logit)[2])

    return grad

end

function constructSubGradients(res::Vector{L}, xMat::Matrix{M}, observationIDs::Vector{O}, parameters::Vector{Symbol}, randomParameters::Dict{Symbol, String}, normalDraws::Vector{L}) where {L<:Real, M<:Real, O<:Any}



    for i in unique(observationIDs)
        tempInds = observationIDs.==i
        tempRes = res[tempInds, :]
        tempXMat = xMat[tempInds, :]
        tempNormalDrawMat = reshape(repeat(normalDraws, size(tempXMat)[1]), :, 2)
    end

end
function getDraws(k::Int, n::Int; RandomDraws::Bool=false)

    qmc = pyimport("scipy.stats.qmc")

    sobol = qmc.Sobol(k, seed=1995)
    draws = sobol.random(n)

    return transpose(draws)

end

function getDraws(randomParameters::Dict{Symbol, String}, nInterDraws::Int; RandomDraws::Bool=false)

    return Matrix(getDraws(length(randomParameters), nInterDraws; RandomDraws=RandomDraws))

end

function getDraws(randomParameters::Nothing, nInterDraws::Int; RandomDraws::Bool=false)

    return nothing

end

function getSimulatedParameters(parameters::Vector{Symbol}, randomParameters::Dict{Symbol, String}, randomDraws::Matrix{T}, theta::T...) where{T<:Real}

    numDraws = size(randomDraws)[2]
    numFixed = length(parameters)
    numRandom = length(randomParameters)*2

    fixedParams = theta[1:numFixed]

    params = reshape(repeat([fixedParams...], numDraws), numFixed, :)

    i = 1
    tempParams::Vector{T} = quantile.(Normal(), randomDraws[i, :])
    tempDist::String = "N"
    tempMean::T = theta[numFixed+i]
    tempSD::T = theta[numFixed+i+length(randomParameters)]
    for param in keys(randomParameters)
        tempDist = randomParameters[param]
        tempMean = theta[numFixed+i]
        tempSD = theta[numFixed+i+length(randomParameters)]

        tempParams = quantile.(Normal(tempMean, abs(tempSD)), randomDraws[i, :])
        params = vcat(params, transpose(tempParams))

        i+=1
    end

    return params

end

function getStandardNormalDraws(randomDraws::Matrix{T}) where{T<:Real}

    return quantile.(Normal(), randomDraws)

end

function getStandardNormalDraws(randomDraws::Nothing)

    return nothing

end

function getStartingValues(modelInfo::ModelInfo)

    if(modelInfo.RandomParameters==nothing)

        return repeat([0.], length(modelInfo.Parameters))
    else
        tempFixedParameters = deepcopy(modelInfo.Parameters)

        for tempVar in keys(modelInfo.RandomParameters)
            push!(tempFixedParameters, tempVar)
        end

        simpleModelInfo = ModelInfo(modelInfo.Data, modelInfo.ChoiceColumn, tempFixedParameters, modelInfo.ChoiceSetIDColumn, nothing, nothing, modelInfo.PanelIDColumn, modelInfo.ObservationIDColumn, modelInfo.DataStructure, modelInfo.ModelSpace, modelInfo.Name, modelInfo.InterDrawCount, modelInfo.WeightColumn, modelInfo.TimeCreated, modelInfo.PreferenceSearchBounds)
        tempRes = EstimateModel(simpleModelInfo; post=false)
        numRandParams::Int = length(modelInfo.RandomParameters)

        startSigmas = repeat([0.3], numRandParams)

        startValues = tempRes["coef"]

        push!(startValues, startSigmas...)

        return startValues

    end
end