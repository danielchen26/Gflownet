# SQLite Persistence for Molecular GFlowNet
#
# Thread-safe database operations using ReentrantLock + WAL mode.
# Training runs in @async while Oxygen handles HTTP concurrently.

using SQLite
using UUIDs: uuid4
using Dates: DateTime, now, format

# ============================================
# Global State (thread-safe)
# ============================================

const DB_LOCK = ReentrantLock()
const MOL_DB = Ref{Union{SQLite.DB, Nothing}}(nothing)

# ============================================
# Initialization
# ============================================

"""
    init_database!(path::String)

Initialize SQLite database with schema and WAL mode for concurrent access.
"""
function init_database!(path::String)
    # Ensure directory exists
    mkpath(dirname(path))

    db = SQLite.DB(path)

    # WAL mode for concurrent reads during training
    SQLite.execute(db, "PRAGMA journal_mode=WAL")
    SQLite.execute(db, "PRAGMA busy_timeout=5000")
    SQLite.execute(db, "PRAGMA synchronous=NORMAL")

    # Create tables
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS training_sessions (
            id TEXT PRIMARY KEY,
            domain_type TEXT NOT NULL,
            config_json TEXT,
            started_at TEXT,
            completed_at TEXT,
            total_iterations INTEGER,
            final_loss REAL,
            molecule_count INTEGER DEFAULT 0
        )
    """)

    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS molecules (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            smiles TEXT NOT NULL,
            canonical_smiles TEXT,
            reward REAL,
            generation_step INTEGER,
            method TEXT DEFAULT 'fragment_gflownet',
            created_at TEXT DEFAULT (datetime('now')),
            molecular_weight REAL,
            logp REAL,
            qed REAL,
            sa_score REAL,
            tpsa REAL,
            hbd INTEGER,
            hba INTEGER,
            rotatable_bonds INTEGER,
            num_rings INTEGER,
            num_aromatic_rings INTEGER,
            formula TEXT,
            svg_2d TEXT,
            fingerprint BLOB,
            FOREIGN KEY(session_id) REFERENCES training_sessions(id)
        )
    """)

    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS idx_molecules_session ON molecules(session_id)")
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS idx_molecules_reward ON molecules(reward DESC)")
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS idx_molecules_qed ON molecules(qed DESC)")
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS idx_molecules_smiles ON molecules(smiles)")

    MOL_DB[] = db
    @info "Molecule database initialized" path=path
    return db
end

# ============================================
# Thread-Safe Execution
# ============================================

"""Execute a function with the database lock held."""
function db_execute(f::Function)
    db = MOL_DB[]
    db === nothing && error("Database not initialized. Call init_database! first.")
    lock(DB_LOCK) do
        f(db)
    end
end

# ============================================
# Session Operations
# ============================================

"""Record a new training session."""
function db_create_session!(session_id::String, domain_type::String, config::Dict)
    db_execute() do db
        SQLite.execute(db,
            "INSERT OR REPLACE INTO training_sessions (id, domain_type, config_json, started_at) VALUES (?, ?, ?, ?)",
            [session_id, domain_type, JSON3.write(config), format(now(), "yyyy-mm-dd HH:MM:SS")]
        )
    end
end

"""Update session completion status."""
function db_complete_session!(session_id::String, final_loss::Float64, molecule_count::Int)
    db_execute() do db
        SQLite.execute(db,
            "UPDATE training_sessions SET completed_at = ?, final_loss = ?, molecule_count = ? WHERE id = ?",
            [format(now(), "yyyy-mm-dd HH:MM:SS"), final_loss, molecule_count, session_id]
        )
    end
end

"""List all training sessions."""
function db_list_sessions(; limit::Int=20, offset::Int=0)
    db_execute() do db
        results = SQLite.DBInterface.execute(db,
            "SELECT * FROM training_sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
            [limit, offset]
        ) |> SQLite.rowtable
        return [Dict(pairs(row)) for row in results]
    end
end

# ============================================
# Molecule Operations
# ============================================

"""Store a molecule in the database."""
function db_store_molecule!(mol::Dict, session_id::String)
    props = get(mol, "properties", Dict())
    fp = get(mol, "fingerprint", nothing)
    fp_blob = fp !== nothing ? Vector{UInt8}(reinterpret(UInt8, Float32.(fp))) : nothing

    db_execute() do db
        SQLite.execute(db, """
            INSERT OR IGNORE INTO molecules
            (id, session_id, smiles, reward, generation_step, method,
             molecular_weight, logp, qed, sa_score, tpsa, hbd, hba,
             rotatable_bonds, num_rings, num_aromatic_rings, formula,
             svg_2d, fingerprint)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            get(mol, "id", string(uuid4())),
            session_id,
            get(mol, "smiles", ""),
            get(mol, "reward", 0.0),
            get(mol, "generation_step", 0),
            get(mol, "method", "fragment_gflownet"),
            get(props, "molecular_weight", nothing),
            get(props, "logp", nothing),
            get(props, "qed", nothing),
            get(props, "synthetic_accessibility", nothing),
            get(props, "tpsa", nothing),
            get(props, "hbd", nothing),
            get(props, "hba", nothing),
            get(props, "rotatable_bonds", nothing),
            get(props, "num_rings", nothing),
            get(props, "num_aromatic_rings", nothing),
            get(props, "formula", nothing),
            get(mol, "svg_2d", nothing),
            fp_blob,
        ])
    end
end

"""Batch store multiple molecules."""
function db_store_molecules!(molecules::Vector{Dict}, session_id::String)
    db_execute() do db
        SQLite.transaction(db) do
            for mol in molecules
                props = get(mol, "properties", Dict())
                fp = get(mol, "fingerprint", nothing)
                fp_blob = fp !== nothing ? Vector{UInt8}(reinterpret(UInt8, Float32.(fp))) : nothing

                SQLite.execute(db, """
                    INSERT OR IGNORE INTO molecules
                    (id, session_id, smiles, reward, generation_step, method,
                     molecular_weight, logp, qed, sa_score, tpsa, hbd, hba,
                     rotatable_bonds, num_rings, num_aromatic_rings, formula,
                     svg_2d, fingerprint)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    get(mol, "id", string(uuid4())),
                    session_id,
                    get(mol, "smiles", ""),
                    get(mol, "reward", 0.0),
                    get(mol, "generation_step", 0),
                    get(mol, "method", "fragment_gflownet"),
                    get(props, "molecular_weight", nothing),
                    get(props, "logp", nothing),
                    get(props, "qed", nothing),
                    get(props, "synthetic_accessibility", nothing),
                    get(props, "tpsa", nothing),
                    get(props, "hbd", nothing),
                    get(props, "hba", nothing),
                    get(props, "rotatable_bonds", nothing),
                    get(props, "num_rings", nothing),
                    get(props, "num_aromatic_rings", nothing),
                    get(props, "formula", nothing),
                    get(mol, "svg_2d", nothing),
                    fp_blob,
                ])
            end
        end
    end
end

"""Query molecules with filtering and pagination."""
function db_query_molecules(;
    session_id::Union{String,Nothing}=nothing,
    min_reward::Union{Float64,Nothing}=nothing,
    min_qed::Union{Float64,Nothing}=nothing,
    sort_by::String="reward",
    sort_order::String="DESC",
    limit::Int=20,
    offset::Int=0
)
    conditions = String[]
    params = Any[]

    if session_id !== nothing
        push!(conditions, "session_id = ?")
        push!(params, session_id)
    end
    if min_reward !== nothing
        push!(conditions, "reward >= ?")
        push!(params, min_reward)
    end
    if min_qed !== nothing
        push!(conditions, "qed >= ?")
        push!(params, min_qed)
    end

    where_clause = isempty(conditions) ? "" : "WHERE " * join(conditions, " AND ")

    # Sanitize sort column
    valid_sorts = ["reward", "qed", "molecular_weight", "logp", "generation_step", "created_at"]
    sort_col = sort_by in valid_sorts ? sort_by : "reward"
    sort_dir = uppercase(sort_order) == "ASC" ? "ASC" : "DESC"

    query = "SELECT * FROM molecules $where_clause ORDER BY $sort_col $sort_dir LIMIT ? OFFSET ?"
    push!(params, limit, offset)

    count_query = "SELECT COUNT(*) as total FROM molecules $where_clause"
    count_params = params[1:end-2]

    db_execute() do db
        rows = SQLite.DBInterface.execute(db, query, params) |> SQLite.rowtable
        total_result = SQLite.DBInterface.execute(db, count_query, count_params) |> SQLite.rowtable
        total = isempty(total_result) ? 0 : total_result[1].total

        molecules = [_row_to_mol_dict(row) for row in rows]
        return (molecules=molecules, total=total)
    end
end

"""Convert a database row to a molecule Dict matching the adapter format."""
function _row_to_mol_dict(row)
    # Reconstruct fingerprint from BLOB if present
    fp = if row.fingerprint !== nothing && !isempty(row.fingerprint)
        try
            reinterpret(Float32, Vector{UInt8}(row.fingerprint)) |> Vector{Float64}
        catch
            nothing
        end
    else
        nothing
    end

    Dict(
        "id" => row.id,
        "smiles" => row.smiles,
        "reward" => row.reward,
        "generation_step" => row.generation_step,
        "method" => row.method,
        "properties" => Dict(
            "molecular_weight" => something(row.molecular_weight, 0.0),
            "logp" => something(row.logp, 0.0),
            "qed" => something(row.qed, 0.0),
            "synthetic_accessibility" => something(row.sa_score, 0.0),
            "tpsa" => something(row.tpsa, 0.0),
            "rotatable_bonds" => something(row.rotatable_bonds, 0),
            "hbd" => something(row.hbd, 0),
            "hba" => something(row.hba, 0),
            "num_rings" => something(row.num_rings, 0),
            "num_aromatic_rings" => something(row.num_aromatic_rings, 0),
            "formula" => something(row.formula, ""),
        ),
        "svg_2d" => row.svg_2d,
        "fingerprint" => fp,
        "created_at" => something(row.created_at, ""),
    )
end

"""Get total molecule count."""
function db_molecule_count(; session_id::Union{String,Nothing}=nothing)
    db_execute() do db
        if session_id !== nothing
            result = SQLite.DBInterface.execute(db,
                "SELECT COUNT(*) as total FROM molecules WHERE session_id = ?",
                [session_id]
            ) |> SQLite.rowtable
        else
            result = SQLite.DBInterface.execute(db,
                "SELECT COUNT(*) as total FROM molecules"
            ) |> SQLite.rowtable
        end
        return isempty(result) ? 0 : result[1].total
    end
end
