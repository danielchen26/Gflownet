# Edit Operators — explicit search actions for hierarchical edit GFlowNets
#
# Minimal first batches should use a truthful trusted kernel:
# - mutate
# - crossover
# - terminate
#
# Fragment-style operators remain experimental until validated.

using Random

abstract type AbstractEditOperator end

struct MutateOperator <: AbstractEditOperator end
struct CrossoverOperator <: AbstractEditOperator end
struct AddFragmentOperator <: AbstractEditOperator end
struct ReplaceFragmentOperator <: AbstractEditOperator end
struct DeleteFragmentOperator <: AbstractEditOperator end
struct TerminateOperator <: AbstractEditOperator end

"""
    EditProposal

Concrete edit proposal produced by an operator.
"""
struct EditProposal
    operator::Symbol
    parent_smiles::String
    partner_smiles::Union{Nothing,String}
    child_smiles::String
    metadata::Dict{String,Any}
end

trusted_edit_operators(; allow_crossover::Bool=true) = allow_crossover ? Symbol[:mutate, :terminate, :crossover] : Symbol[:mutate, :terminate]
experimental_fragment_operators() = Symbol[:add_fragment, :replace_fragment, :delete_fragment]

"""
    available_edit_operators(; allow_crossover=true, allow_fragment_ops=false)

Return the operator menu for the current baseline.
By default, only the trusted kernel is enabled.
"""
function available_edit_operators(; allow_crossover::Bool=true,
                                  allow_fragment_ops::Bool=false)
    ops = trusted_edit_operators(; allow_crossover=allow_crossover)
    allow_fragment_ops && append!(ops, experimental_fragment_operators())
    return ops
end

function _raw_edit_proposals(parent_smiles::String, operator::Symbol, vocab;
                             partner_smiles::Union{Nothing,String}=nothing,
                             max_candidates::Int=8)
    proposals = EditProposal[]

    if operator == :terminate
        push!(proposals, EditProposal(:terminate, parent_smiles, partner_smiles, parent_smiles, Dict("kind" => "identity")))
        return proposals
    elseif operator == :mutate
        children = smiles_mutate_rdkit(parent_smiles; n_mutations=max_candidates)
        for child in children
            push!(proposals, EditProposal(:mutate, parent_smiles, nothing, child, Dict("kind" => "atom_substitution")))
        end
        return proposals
    elseif operator == :crossover
        isnothing(partner_smiles) && return proposals
        children = smiles_crossover_rdkit(parent_smiles, partner_smiles)
        for child in children[1:min(end, max_candidates)]
            push!(proposals, EditProposal(:crossover, parent_smiles, partner_smiles, child, Dict("kind" => "brics_crossover")))
        end
        return proposals
    elseif operator == :add_fragment
        # EXPERIMENTAL proxy only — excluded from default operator menu.
        children = smiles_crossover_rdkit(parent_smiles, parent_smiles)
        for child in children[1:min(end, max_candidates)]
            push!(proposals, EditProposal(:add_fragment, parent_smiles, nothing, child, Dict("kind" => "fragment_add_proxy")))
        end
        return proposals
    elseif operator == :replace_fragment
        # EXPERIMENTAL proxy only — excluded from default operator menu.
        children = smiles_crossover_rdkit(parent_smiles, parent_smiles)
        for child in children[1:min(end, max_candidates)]
            push!(proposals, EditProposal(:replace_fragment, parent_smiles, nothing, child, Dict("kind" => "fragment_replace_proxy")))
        end
        return proposals
    elseif operator == :delete_fragment
        # EXPERIMENTAL proxy only — excluded from default operator menu.
        tokens = encode(vocab, parent_smiles)
        for _ in 1:max_candidates
            mutated_tokens = smiles_mutate_tokens(tokens, vocab; n_mutations=1)
            child = decode(vocab, mutated_tokens)
            push!(proposals, EditProposal(:delete_fragment, parent_smiles, nothing, child, Dict("kind" => "delete_proxy_token_mutation")))
        end
        return proposals
    end

    return proposals
end

function _sanitize_edit_proposals(raw_proposals::Vector{EditProposal}, parent_smiles::String)
    diagnostics = Dict{String,Any}(
        "raw_candidate_count" => length(raw_proposals),
        "duplicate_candidate_count" => 0,
        "empty_child_count" => 0,
        "self_child_count" => 0,
        "unique_valid_count" => 0,
    )

    filtered = EditProposal[]
    seen = Set{String}()

    canonical_parent = canonicalize_smiles_identity(parent_smiles)

    for proposal in raw_proposals
        child = strip(proposal.child_smiles)
        if isempty(child)
            diagnostics["empty_child_count"] = Int(diagnostics["empty_child_count"]) + 1
            continue
        end
        canonical_child = canonicalize_smiles_identity(child)
        if isempty(canonical_child)
            diagnostics["empty_child_count"] = Int(diagnostics["empty_child_count"]) + 1
            continue
        end
        if proposal.operator != :terminate && canonical_child == canonical_parent
            diagnostics["self_child_count"] = Int(diagnostics["self_child_count"]) + 1
            continue
        end
        if canonical_child in seen
            diagnostics["duplicate_candidate_count"] = Int(diagnostics["duplicate_candidate_count"]) + 1
            continue
        end

        push!(filtered, EditProposal(
            proposal.operator,
            proposal.parent_smiles,
            proposal.partner_smiles,
            canonical_child,
            copy(proposal.metadata),
        ))
        push!(seen, canonical_child)
    end

    diagnostics["unique_valid_count"] = length(filtered)
    return filtered, diagnostics
end

"""
    propose_edit(parent_smiles, operator, vocab; partner_smiles=nothing, max_candidates=8)

Generate concrete child proposals for a chosen operator.
Current implementation reuses validated Genetic-GFN utilities where possible.
"""
function propose_edit(parent_smiles::String, operator::Symbol, vocab;
                      partner_smiles::Union{Nothing,String}=nothing,
                      max_candidates::Int=8)
    proposals, _ = propose_edit_with_diagnostics(parent_smiles, operator, vocab;
        partner_smiles=partner_smiles,
        max_candidates=max_candidates)
    return proposals
end

"""
    propose_edit_with_diagnostics(parent_smiles, operator, vocab; ...)

Generate proposals and return proposal-level diagnostics before any frontier-side
cached-child filtering happens.
"""
function propose_edit_with_diagnostics(parent_smiles::String, operator::Symbol, vocab;
                                       partner_smiles::Union{Nothing,String}=nothing,
                                       max_candidates::Int=8)
    raw = _raw_edit_proposals(parent_smiles, operator, vocab;
        partner_smiles=partner_smiles,
        max_candidates=max_candidates)
    return _sanitize_edit_proposals(raw, parent_smiles)
end

"""
    choose_partner(snapshot, parent_smiles; prefer_cross_scaffold, prefer_target_scaffold, diversity_bonus)

Select a crossover partner for BRICS fragment exchange. For effective crossover,
we want partners that are high-reward but structurally *different* from the parent
(cross-scaffold) — this produces novel fragment combinations. Partners sharing
the parent's scaffold tend to produce redundant offspring. Target-scaffold partners
get a bonus for structural task convergence.
"""
function choose_partner(snapshot::FrontierSnapshot, parent_smiles::String;
                        prefer_cross_scaffold::Bool=true,
                        prefer_target_scaffold::Bool=true,
                        diversity_bonus::Float64=1.5,
                        deterministic::Bool=false)
    parent_scaffold = get_scaffold(parent_smiles)
    candidates = [e for e in snapshot.entries if e.smiles != parent_smiles]
    isempty(candidates) && return nothing

    scores = Float64[]
    for candidate in candidates
        score = max(candidate.reward, 1e-8)
        if prefer_target_scaffold && !isnothing(snapshot.target_scaffold) &&
               !isempty(candidate.scaffold) && candidate.scaffold == snapshot.target_scaffold
            score *= 2.0
        elseif prefer_cross_scaffold && !isempty(parent_scaffold) && !isempty(candidate.scaffold) &&
               candidate.scaffold != parent_scaffold
            score *= diversity_bonus
        end
        push!(scores, score)
    end

    idx = deterministic ? argmax(scores) : _sample_partner_index(scores)
    return candidates[idx].smiles
end

function _sample_partner_index(scores::Vector{Float64})
    total = sum(scores)
    probs = total > 0 ? scores ./ total : fill(1.0 / length(scores), length(scores))
    cumulative = cumsum(probs)
    r = rand()
    return clamp(searchsortedfirst(cumulative, r), 1, length(scores))
end

function unique_child_proposals(proposals::Vector{EditProposal}; parent_smiles::String="")
    filtered, _ = _sanitize_edit_proposals(proposals, parent_smiles)
    return filtered
end
