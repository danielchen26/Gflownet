# Metal GPU Acceleration for GFlowNet
#
# Fixes Metal.jl v1.9.2 + Julia 1.12 LinearAlgebra.mul! dispatch issue.
# Metal.MPS.matmul! works but the standard mul!/gemm! path dispatches to
# CPU BLAS for MtlMatrix{PrivateStorage}, crashing with:
#   "cannot take the CPU address of a MtlMatrix{Float32, Metal.PrivateStorage}"
#
# This module defines the missing mul! overrides that redirect to MPS.
# Usage: call `init_metal_accel!()` early in your script.
#
# Status (Metal.jl v1.9.2, Julia 1.12.4, M4 Max, March 2026):
# ---------------------------------------------------------------
# WORKING:
#   - Raw matmul (MPS): 3-5x faster than CPU/AMX at batch>=256
#   - Dense(identity) forward + Zygote backward
#   - GRUCell forward + Zygote backward
#   - Embedding lookup
#
# NOT WORKING:
#   - Dense(tanh/relu/etc.) — LuxLib's fused_dense_bias_activation compiles
#     `tanh_fast ∘ +` as a Metal kernel → InvalidIRError (ComposedFunction)
#   - Full CAFE-GFN pipeline (output_dense uses Dense(tanh) internally)
#
# RECOMMENDATION:
#   CPU with AppleAccelerate/AMX is the optimal path for CAFE-GFN training.
#   At batch=256: ~14K tok/s, ~15 min/epoch for ZINC 250K, ~3 TFLOPS.
#   GPU overhead from kernel dispatch negates raw matmul advantage for our
#   GRU-sized operations.

const _metal_available = Ref(false)
const _metal_accel_initialized = Ref(false)

"""
    init_metal_accel!()

Initialize Metal GPU acceleration by defining missing `LinearAlgebra.mul!`
overrides for `MtlMatrix`. Call once at startup.

This fixes Metal.jl v1.9.2's missing dispatch: `LinearAlgebra.mul!` for
`MtlMatrix{PrivateStorage}` now routes to `Metal.MPS.matmul!` instead of
falling back to CPU BLAS (which crashes).

Supports all combinations: C = A*B, C = A'*B, C = A*B', plus 5-arg αβ form.

Returns `true` if Metal is available and overrides were installed, `false` otherwise.
"""
function init_metal_accel!()
    if _metal_accel_initialized[]
        return _metal_available[]
    end

    _metal_accel_initialized[] = true

    try
        @eval begin
            using Metal

            if Metal.functional()
                # Union type for MtlMatrix and its Adjoint/Transpose wrappers
                const MtlMatOrAdj{T} = Union{
                    MtlMatrix{T},
                    Adjoint{T, <:MtlMatrix{T}},
                    Transpose{T, <:MtlMatrix{T}}
                }

                # Helper: extract transpose flag and raw array
                _mtl_unpack(A::MtlMatrix{T}) where T = ('N', A)
                _mtl_unpack(A::Adjoint{T, <:MtlMatrix{T}}) where T = ('T', parent(A))
                _mtl_unpack(A::Transpose{T, <:MtlMatrix{T}}) where T = ('T', parent(A))

                # Core MPS-backed matmul
                function _mtl_matmul!(
                    C::MtlMatrix{T}, A::MtlMatOrAdj{T}, B::MtlMatOrAdj{T},
                    alpha::T, beta::T
                ) where {T<:Union{Float16, Float32}}
                    tA, rawA = _mtl_unpack(A)
                    tB, rawB = _mtl_unpack(B)
                    Metal.MPS.matmul!(C, rawA, rawB, alpha, beta, tA == 'T', tB == 'T')
                    return C
                end

                # 3-arg: C = A * B
                function LinearAlgebra.mul!(
                    C::MtlMatrix{T}, A::MtlMatOrAdj{T}, B::MtlMatOrAdj{T}
                ) where {T<:Union{Float16, Float32}}
                    _mtl_matmul!(C, A, B, one(T), zero(T))
                end

                # 5-arg: C = α*A*B + β*C
                function LinearAlgebra.mul!(
                    C::MtlMatrix{T}, A::MtlMatOrAdj{T}, B::MtlMatOrAdj{T},
                    alpha::Number, beta::Number
                ) where {T<:Union{Float16, Float32}}
                    _mtl_matmul!(C, A, B, T(alpha), T(beta))
                end

                # Matrix-vector: c = A*b
                function LinearAlgebra.mul!(
                    c::MtlVector{T}, A::MtlMatOrAdj{T}, b::MtlVector{T}
                ) where {T<:Union{Float16, Float32}}
                    _mtl_matmul!(
                        reshape(c, length(c), 1), A,
                        reshape(b, length(b), 1), one(T), zero(T)
                    )
                    return c
                end

                function LinearAlgebra.mul!(
                    c::MtlVector{T}, A::MtlMatOrAdj{T}, b::MtlVector{T},
                    alpha::Number, beta::Number
                ) where {T<:Union{Float16, Float32}}
                    _mtl_matmul!(
                        reshape(c, length(c), 1), A,
                        reshape(b, length(b), 1), T(alpha), T(beta)
                    )
                    return c
                end

                $_metal_available[] = true
                @info "Metal GPU acceleration initialized" device=Metal.current_device()
            else
                @warn "Metal.jl loaded but not functional"
            end
        end
    catch e
        @debug "Metal.jl not available" exception=e
    end

    return _metal_available[]
end

"""
    metal_available()

Check if Metal GPU acceleration is available and initialized.
"""
metal_available() = _metal_available[]

"""
    to_metal(x)

Move array/params to Metal GPU if available, otherwise return as-is.
"""
function to_metal(x)
    if _metal_available[]
        @eval using MLDataDevices
        dev = @eval gpu_device()
        return x |> dev
    end
    return x
end

"""
    from_metal(x)

Move array/params from Metal GPU to CPU.
"""
function from_metal(x)
    if x isa AbstractArray
        return Array(x)
    end
    return x
end
