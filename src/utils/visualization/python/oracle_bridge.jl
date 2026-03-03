# Oracle Bridge — Julia ↔ Python via PythonCall.jl
#
# Bridges TDC (Therapeutics Data Commons) oracles for target-specific
# molecular activity prediction (DRD2, GSK3B, JNK3, GuacaMol tasks).
#
# Mirrors the rdkit_bridge.jl pattern: lazy init, explicit init function,
# batch evaluation to minimize PythonCall crossings.
#
# NOTE: Do NOT use Threads.@spawn with PythonCall — GIL is not thread-safe
# across Julia threads. All oracle calls must happen on the main thread.

module OracleBridge

using PythonCall

# Lazy-loaded state (same pattern as rdkit_bridge.jl)
const _oracle_instances = Ref{Dict{String,Py}}(Dict{String,Py}())
const _initialized = Ref{Bool}(false)
const _cache_dir = Ref{String}("")

# Full list of PMO benchmark tasks (23 tasks)
const PMO_23_TASKS = [
    "albuterol_similarity", "amlodipine_mpo", "celecoxib_rediscovery",
    "deco_hop", "drd2", "fexofenadine_mpo", "gsk3b",
    "isomers_c7h8n2o2", "isomers_c9h10n2o2pf2cl", "jnk3",
    "median1", "median2", "mestranol_similarity", "osimertinib_mpo",
    "perindopril_mpo", "qed", "ranolazine_mpo", "scaffold_hop",
    "sitagliptin_mpo", "thiothixene_rediscovery", "troglitazone_rediscovery",
    "valsartan_smarts", "zaleplon_mpo"
]

# Bioactivity oracles (subset with pretrained ML models)
const BIOACTIVITY_ORACLES = ["drd2", "gsk3b", "jnk3"]

"""
Ensure pkg_resources is available in the Python environment.
conda-forge setuptools >= 71 no longer bundles pkg_resources as a top-level module.
TDC uses it for version checks and resource paths.
Writes a shim file to site-packages if needed.
"""
function _ensure_pkg_resources()
    try
        pyimport("pkg_resources")
    catch
        @info "Creating pkg_resources shim (conda-forge setuptools >= 71 removed it)..."
        # Find site-packages directory
        site = pyimport("site")
        sp_dirs = pyconvert(Vector{String}, site.getsitepackages())
        sp_dir = sp_dirs[1]

        shim_path = joinpath(sp_dir, "pkg_resources.py")
        if !isfile(shim_path)
            shim_code = """
# pkg_resources shim — created by oracle_bridge.jl
# conda-forge setuptools >= 71 removed pkg_resources as a top-level module.
# TDC uses get_distribution() and resource_filename() at import time.

import importlib.metadata
import importlib.resources
import importlib.util
import os

class _Distribution:
    def __init__(self, name):
        self.version = importlib.metadata.version(name)

def get_distribution(name):
    return _Distribution(name)

def resource_filename(package, resource):
    try:
        ref = importlib.resources.files(package).joinpath(resource)
        return str(ref)
    except Exception:
        spec = importlib.util.find_spec(package)
        if spec and spec.origin:
            return os.path.join(os.path.dirname(spec.origin), resource)
        return resource
"""
            write(shim_path, shim_code)
            @info "Wrote pkg_resources shim" path=shim_path
        end
        pyimport("pkg_resources")
    end
end

"""
Ensure rdkit.six module exists (removed in rdkit 2025+).
TDC imports `from rdkit.six import iteritems` which no longer exists.
Creates a minimal shim in rdkit's package directory.
"""
function _ensure_rdkit_six()
    try
        pyimport("rdkit.six")
    catch
        @info "Creating rdkit.six shim (removed in rdkit 2025+)..."
        rdkit = pyimport("rdkit")
        rdkit_dir = pyconvert(String, rdkit.__path__[0])
        six_path = joinpath(rdkit_dir, "six.py")
        if !isfile(six_path)
            six_code = """
# rdkit.six shim — created by oracle_bridge.jl
# rdkit 2025+ removed the bundled six module. TDC imports iteritems from here.
# In Python 3, dict.items() already returns a view, so iteritems = items.

def iteritems(d):
    return d.items()

def itervalues(d):
    return d.values()

def iterkeys(d):
    return d.keys()
"""
            write(six_path, six_code)
            @info "Wrote rdkit.six shim" path=six_path
        end
        pyimport("rdkit.six")
    end
end

"""
    init_oracles!(names; cache_dir="")

Initialize TDC oracles by name. Downloads model files on first call.
Must be called before any evaluation.

# Arguments
- `names::Vector{String}`: Oracle names (e.g., ["DRD2", "GSK3B"])
- `cache_dir::String`: Directory for model downloads (default: data/tdc_cache/)
"""
function init_oracles!(names::Vector{String}; cache_dir::String="")
    # Set download path to absolute dir (avoid CWD-relative ./oracle/)
    dir = isempty(cache_dir) ? joinpath(@__DIR__, "..", "..", "..", "..", "data", "tdc_cache") : cache_dir
    mkpath(dir)
    _cache_dir[] = dir

    # Ensure compatibility shims exist before importing TDC:
    # 1. pkg_resources — conda-forge setuptools >=71 removed it
    # 2. rdkit.six — rdkit 2025+ removed bundled six module
    _ensure_pkg_resources()
    _ensure_rdkit_six()

    # Lazy install: PyTDC pins rdkit<2024 but works fine with newer versions.
    # Install PyTDC --no-deps (to avoid rdkit conflict), then install the
    # essential runtime deps that TDC oracles need.
    # Uses uv (available in CondaPkg/pixi) to install into the current Python env.
    tdc = try
        pyimport("tdc")
    catch
        @info "PyTDC import failed — installing PyTDC and oracle dependencies..."
        subprocess = pyimport("subprocess")
        sys = pyimport("sys")
        python_path = pyconvert(String, sys.executable)

        # Install PyTDC itself (no deps to avoid rdkit conflict)
        subprocess.check_call(pylist([
            "uv", "pip", "install", "--no-deps", "--python", python_path, "PyTDC"
        ]))
        # Install essential runtime deps for oracle functionality
        # (numpy, pandas, scikit-learn already available via conda)
        subprocess.check_call(pylist([
            "uv", "pip", "install", "--python", python_path,
            "huggingface_hub", "tqdm", "packaging", "networkx", "scipy"
        ]))
        pyimport("tdc")
    end

    for name in names
        # Skip if already loaded
        haskey(_oracle_instances[], name) && continue

        @info "Loading TDC oracle: $name (models download on first use)"
        _oracle_instances[][name] = tdc.Oracle(name=name, path=dir)
    end

    _initialized[] = true
    @info "OracleBridge initialized" n_oracles=length(_oracle_instances[]) cache_dir=dir
end

function _ensure_init()
    _initialized[] || error("OracleBridge not initialized! Call init_oracles! first.")
end

"""
    evaluate(smiles, oracle_name) → Float64

Evaluate a single molecule against a named oracle.
"""
function evaluate(smiles::String, oracle_name::String)::Float64
    _ensure_init()
    haskey(_oracle_instances[], oracle_name) || error("Oracle not loaded: $oracle_name")
    oracle = _oracle_instances[][oracle_name]
    return pyconvert(Float64, oracle(smiles))
end

"""
    evaluate_batch(smiles_list, oracle_name) → Vector{Float64}

Evaluate a batch of molecules against a named oracle.
Falls back to individual calls if batch API fails (numpy compat issues).
"""
function evaluate_batch(smiles_list::Vector{String}, oracle_name::String)::Vector{Float64}
    _ensure_init()
    isempty(smiles_list) && return Float64[]
    haskey(_oracle_instances[], oracle_name) || error("Oracle not loaded: $oracle_name")

    # Try batch first, fall back to individual evaluation
    oracle = _oracle_instances[][oracle_name]
    try
        py_result = oracle(pylist(smiles_list))
        return pyconvert(Vector{Float64}, py_result)
    catch
        # Batch failed (numpy compat) — evaluate individually
        return Float64[evaluate(s, oracle_name) for s in smiles_list]
    end
end

"""
    get_loaded_oracles() → Vector{String}

Return names of currently loaded oracles.
"""
function get_loaded_oracles()::Vector{String}
    return collect(keys(_oracle_instances[]))
end

"""
    is_initialized() → Bool

Check if the oracle bridge has been initialized.
"""
is_initialized() = _initialized[]

"""
    get_all_available_oracles() → Dict

Return categorized list of all available TDC oracles.
"""
function get_all_available_oracles()::Dict{String,Vector{String}}
    return Dict(
        "bioactivity" => BIOACTIVITY_ORACLES,
        "pmo_tasks" => PMO_23_TASKS,
        "all" => unique(vcat(BIOACTIVITY_ORACLES, PMO_23_TASKS)),
    )
end

end # module OracleBridge
