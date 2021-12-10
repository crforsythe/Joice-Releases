#=
JoiceTypeDefinitions:
- Julia version: 1.6
- Author: connorforsythe
- Date: 2021-11-05
=#


abstract type JoiceModel end
abstract type JoiceModelType end
abstract type JoiceModelSpace end
abstract type JoiceDataStructure end
export EstimateModel, f

struct WTP <: JoiceModelSpace end
struct Preference <: JoiceModelSpace end

struct Panel <: JoiceDataStructure end
struct Cross <: JoiceDataStructure end



ColType = Union{String, Symbol, Nothing}
ColsType = Union{Vector{String}, Vector{Symbol}, Nothing, ColType}
RandColsType = Union{Dict{String, String}, Dict{Symbol, String}, Nothing}
NameType = Union{String, Nothing}

struct ModelInfo

    Data::DataFrame
    ChoiceColumn::Symbol
    Parameters::Vector{Symbol}
    ChoiceSetIDColumn::Symbol
    RandomParameters::Union{Dict{Symbol, String}, Nothing}
    RandomDraws::Union{Matrix{T}, Nothing} where {T<:Real}
    PanelIDColumn::Union{Symbol, Nothing}
    ObservationIDColumn::Symbol
    DataStructure::JoiceDataStructure
    ModelSpace::JoiceModelSpace
    Name::Union{String, Nothing}
    InterDrawCount::Int
    WeightColumn::Union{Symbol, Nothing}
    TimeCreated::DateTime
    PreferenceSearchBounds::Vector{T} where {T<:Real}

end


struct LogitPrefModel <: JoiceModel

    Data::DataFrame
    ChoiceColumn::Symbol
    Parameters::Vector{Symbol}
    ChoiceSetIDColumn::Symbol
    RandomParameters::Union{Dict{Symbol, String}, Nothing}
    PanelIDColumn::Union{Symbol, Nothing}
    Space::JoiceModelSpace
    Name::Union{String, Nothing}
    InterDrawCount::Int
    WeightColumn::Union{Symbol, Nothing}
    TimeCreated::DateTime

end
