# Unit tests for Gap 1: Pairwise Tanimoto Diversity Metrics
# Tests for compute_tanimoto_matrix, compute_diversity_stats, compute_scaffold_diversity

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

# RDKit availability comes from the single gate in test/fixtures/molecular.jl
# (loaded by test_setup.jl above), which also logs WHY it is unavailable.
# Previously this read `isdefined(Main, :RDKitBridge)` while sibling files used
# `@isdefined(RDKitBridge)` — three idioms for one condition.
const _rdkit_available = RDKIT_AVAILABLE

@testset "Tanimoto Diversity Metrics" begin

    @testset "Tanimoto matrix — identical fingerprints" begin
        if !_rdkit_available
            @test_skip "RDKitBridge not available"
        else
            # Identical fingerprints should have Tanimoto = 1.0
            fp = Float32[1, 0, 1, 0, 1, 1, 0, 0]
            fps = [fp, copy(fp), copy(fp)]

            sim_matrix = RDKitBridge.compute_tanimoto_matrix(fps)
            @test size(sim_matrix) == (3, 3)

            # Diagonal should be 1.0
            for i in 1:3
                @test sim_matrix[i, i] ≈ 1.0
            end

            # All pairwise should be 1.0 (identical)
            for i in 1:3, j in 1:3
                @test sim_matrix[i, j] ≈ 1.0
            end
        end
    end

    @testset "Tanimoto matrix — orthogonal fingerprints" begin
        if !_rdkit_available
            @test_skip "RDKitBridge not available"
        else
            # Non-overlapping fingerprints should have Tanimoto = 0.0
            fp1 = Float32[1, 1, 0, 0, 0, 0, 0, 0]
            fp2 = Float32[0, 0, 1, 1, 0, 0, 0, 0]
            fps = [fp1, fp2]

            sim_matrix = RDKitBridge.compute_tanimoto_matrix(fps)
            @test size(sim_matrix) == (2, 2)
            @test sim_matrix[1, 1] ≈ 1.0
            @test sim_matrix[2, 2] ≈ 1.0
            @test sim_matrix[1, 2] ≈ 0.0
            @test sim_matrix[2, 1] ≈ 0.0
        end
    end

    @testset "Tanimoto matrix — symmetry" begin
        if !_rdkit_available
            @test_skip "RDKitBridge not available"
        else
            fp1 = Float32[1, 0, 1, 1, 0, 0, 1, 0]
            fp2 = Float32[0, 1, 1, 0, 1, 0, 1, 0]
            fp3 = Float32[1, 1, 0, 0, 0, 1, 0, 1]
            fps = [fp1, fp2, fp3]

            sim_matrix = RDKitBridge.compute_tanimoto_matrix(fps)
            for i in 1:3, j in 1:3
                @test sim_matrix[i, j] ≈ sim_matrix[j, i]  # Symmetric
            end
        end
    end

    @testset "Tanimoto matrix — empty input" begin
        if !_rdkit_available
            @test_skip "RDKitBridge not available"
        else
            sim_matrix = RDKitBridge.compute_tanimoto_matrix(Vector{Float32}[])
            @test size(sim_matrix) == (0, 0)
        end
    end

    @testset "diversity_stats — diverse set" begin
        if !_rdkit_available
            @test_skip "RDKitBridge not available"
        else
            # Create diverse fingerprints
            fp1 = Float32[1, 0, 0, 0, 0, 0, 0, 0]
            fp2 = Float32[0, 1, 0, 0, 0, 0, 0, 0]
            fp3 = Float32[0, 0, 1, 0, 0, 0, 0, 0]
            fp4 = Float32[0, 0, 0, 1, 0, 0, 0, 0]
            fps = [fp1, fp2, fp3, fp4]

            stats = RDKitBridge.compute_diversity_stats(fps)

            @test haskey(stats, "mean_pairwise")
            @test haskey(stats, "internal_diversity_1")
            @test haskey(stats, "internal_diversity_2")
            @test haskey(stats, "median_nn_distance")
            @test haskey(stats, "n_molecules")

            # Orthogonal fps → mean pairwise should be 0.0, IntDiv1 should be 1.0
            @test stats["mean_pairwise"] ≈ 0.0
            @test stats["internal_diversity_1"] ≈ 1.0
            @test stats["n_molecules"] == 4
        end
    end

    @testset "diversity_stats — identical set" begin
        if !_rdkit_available
            @test_skip "RDKitBridge not available"
        else
            fp = Float32[1, 1, 1, 0, 0, 0, 0, 0]
            fps = [copy(fp) for _ in 1:5]

            stats = RDKitBridge.compute_diversity_stats(fps)

            @test stats["mean_pairwise"] ≈ 1.0
            @test stats["internal_diversity_1"] ≈ 0.0
            @test stats["n_molecules"] == 5
        end
    end

    @testset "diversity_stats — single molecule" begin
        if !_rdkit_available
            @test_skip "RDKitBridge not available"
        else
            fp = Float32[1, 0, 1, 0]
            stats = RDKitBridge.compute_diversity_stats([fp])

            # With only 1 molecule, should return default values
            @test stats["n_molecules"] == 1
            @test stats["internal_diversity_1"] == 1.0
        end
    end

    @testset "scaffold diversity — known molecules" begin
        if !_rdkit_available
            @test_skip "RDKitBridge not available"
        else
            # Benzene and naphthalene have different scaffolds
            smiles = ["c1ccccc1", "c1ccc2ccccc2c1", "c1ccccc1"]  # benzene, naphthalene, benzene again

            scaffold_stats = RDKitBridge.compute_scaffold_diversity(smiles)

            @test haskey(scaffold_stats, "n_unique_scaffolds")
            @test haskey(scaffold_stats, "scaffold_entropy")
            @test haskey(scaffold_stats, "scaffold_distribution")

            # Should find 2 unique scaffolds (benzene and naphthalene)
            @test scaffold_stats["n_unique_scaffolds"] == 2
            @test scaffold_stats["scaffold_entropy"] > 0.0
        end
    end

    @testset "scaffold diversity — empty input" begin
        if !_rdkit_available
            @test_skip "RDKitBridge not available"
        else
            scaffold_stats = RDKitBridge.compute_scaffold_diversity(String[])
            @test scaffold_stats["n_unique_scaffolds"] == 0
            @test scaffold_stats["scaffold_entropy"] == 0.0
        end
    end

    @testset "nearest neighbors" begin
        if !_rdkit_available
            @test_skip "RDKitBridge not available"
        else
            fp1 = Float32[1, 1, 0, 0, 0, 0, 0, 0]
            fp2 = Float32[1, 0, 0, 0, 0, 0, 0, 0]  # Similar to fp1
            fp3 = Float32[0, 0, 0, 0, 1, 1, 0, 0]  # Different
            fps = [fp1, fp2, fp3]
            ids = ["mol_1", "mol_2", "mol_3"]

            nn = RDKitBridge.compute_nearest_neighbors(fps, ids; k=1)

            @test length(nn) >= 3  # At least 1 neighbor per molecule
            # mol_1's nearest neighbor should be mol_2 (highest similarity)
            mol1_nn = filter(d -> d["id"] == "mol_1", nn)
            @test length(mol1_nn) >= 1
            @test mol1_nn[1]["neighbor_id"] == "mol_2"
        end
    end
end
