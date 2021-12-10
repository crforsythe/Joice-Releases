#=
JoiceTypeConstructorFunctions:
- Julia version: 1.6
- Author: connorforsythe
- Date: 2021-11-05
=#

export ConstructModelInfo


function checkModelInfo(Data::DataFrame, ChoiceColumn::Symbol, Parameters::Vector{Symbol}, ChoiceSetIDColumn::Symbol, RandomParameters::RandColsType=nothing, PanelIDColumn::ColsType=nothing, WeightColumn::ColsType=nothing)

    dataCols = Symbol.(names(Data))
    missingCols = []
    if !(ChoiceColumn in dataCols)
        push!(missingCols, ChoiceColumn)
    end
    if !(ChoiceSetIDColumn in dataCols)
        push!(missingCols, ChoiceSetIDColumn)
    end
    if !(PanelIDColumn in dataCols) && (PanelIDColumn!=nothing)
        push!(missingCols, PanelIDColumn)
    end


    for tempSymbol in Parameters
        if !(tempSymbol in dataCols)
            push!(missingCols, tempSymbol)
        end
    end

    if RandomParameters!=nothing
        for tempSymbol in keys(RandomParameters)
            if !(tempSymbol in dataCols)
                push!(missingCols, tempSymbol)
            end
        end
    end

    if length(missingCols)>0
        error("You have specified the following cols that aren't in the provided dataset: $missingCols")
    end
end

function convertParameterColumns(x::Vector{String}) return Symbol.(x) end
function convertParameterColumns(x::Vector{Symbol}) return x end
function convertParameterColumns(x::Symbol) return [x] end
function convertParameterColumns(x::String) return convertParameterColumns(Symbol(x)) end
function convertParameterColumns(x::Nothing) return x end
function convertParameterColumn(x::String) return Symbol(x) end
function convertParameterColumn(x::Symbol) return x end
function convertParameterColumn(x::Nothing) return x end
function convertRandomParameterColumns(x::Dict{Symbol, String}) return x end
function convertRandomParameterColumns(x::Dict{String, String}) return Dict(zip(convertParameterColumns([keys(x)...]), values(x))) end
function convertRandomParameterColumns(x::Nothing) return x end

function ConstructModelInfo(Data::DataFrame, ChoiceColumn::ColType, Parameters::ColsType, ChoiceSetIDColumn::ColType; RandomParameters::RandColsType=nothing, PanelIDColumn::ColType=nothing,
    Space::String="preference", Name::NameType=nothing, InterDrawCount::Int=2048, WeightColumn::ColType=nothing)


    TimeCreated = Dates.now()

    ChoiceColumn = convertParameterColumn(ChoiceColumn)
    Parameters = convertParameterColumns(Parameters)
    ChoiceSetIDColumn = convertParameterColumn(ChoiceSetIDColumn)
    WeightColumn = convertParameterColumn(WeightColumn)
    PanelIDColumn = convertParameterColumn(PanelIDColumn)
    RandomParameters = convertRandomParameterColumns(RandomParameters)
    RandomDraws = getDraws(RandomParameters, InterDrawCount)
    typedSpace::JoiceModelSpace = Preference()


    colsToKeep = copy(Parameters)

    appendParamNames!(colsToKeep, RandomParameters)

    push!(colsToKeep, ChoiceColumn)
    push!(colsToKeep, ChoiceSetIDColumn)

    if(lowercase(Space)=="wtp")
        typedSpace = WTP()
    end

    typedDataStructure::JoiceDataStructure = Panel()
    ObservationColumn::Symbol = ChoiceSetIDColumn
    if(PanelIDColumn==nothing)
        typedDataStructure = Cross()
    else
        push!(colsToKeep, PanelIDColumn)
        ObservationColumn = PanelIDColumn
    end


    if(WeightColumn!=nothing)
        push!(colsToKeep, WeightColumn)
    end

    checkModelInfo(Data, ChoiceColumn, Parameters, ChoiceSetIDColumn, RandomParameters, PanelIDColumn, WeightColumn)
    PreferenceSearchBounds = [-1, 1]

    println("Model being constructed with the following properties:")
    if(PanelIDColumn!=nothing)
        println("Panel Variable: $PanelIDColumn")
    end
    println("Choice Set Variable: $ChoiceSetIDColumn")
    println("Choice Indicator Variable: $ChoiceColumn")
    println("Fixed Parameters: $Parameters")
    if(RandomParameters!=nothing)
        println("Random Parameters: $RandomParameters")
    end
    print("Data has ")
    print(size(Data)[1])
    println(" rows")

    cleanDataFrame = cleanData(Data, colsToKeep, ChoiceSetIDColumn, PanelIDColumn)

    r = ModelInfo(cleanDataFrame, ChoiceColumn, Parameters, :choiceSetID, RandomParameters, RandomDraws,
                    PanelIDColumn, ObservationColumn, typedDataStructure, typedSpace, Name, InterDrawCount,
                    WeightColumn, TimeCreated, PreferenceSearchBounds)

    return r
end


function cleanData(data::DataFrame, colsToKeep::Vector{Symbol}, choiceSetIDColumn::Symbol, panelIDColumn::Symbol)

    data = data[:, colsToKeep]

    #Construct new id to allow for repeated choice set ids given
    maxChoiceSetID = max(data[:, choiceSetIDColumn]...)
    panelIDMultiplier = 10^(ceil(log10(maxChoiceSetID)))

    data[:, :choiceSetID] .= (data[:, panelIDColumn].*panelIDMultiplier).+data[:, choiceSetIDColumn]

    return data

end

function cleanData(data::DataFrame, colsToKeep::Vector{Symbol}, choiceSetIDColumn::Symbol, panelIDColumn::Nothing)

    data = data[:, colsToKeep]

    data[:, :choiceSetID] .= data[:, choiceSetIDColumn]

    return data

end

function appendParamNames!(currentVect::Vector{Symbol}, randomParameters::Nothing) end

function appendParamNames!(currentVect::Vector{Symbol}, randomParameters::Dict{Symbol, String})
    for tempCol in keys(randomParameters)
        push!(currentVect, tempCol)
    end
end