// API Service for GFlowNet Real Training Visualization
// Connects to the v2 backend endpoints implemented in unified_server.jl

import axios from '../lib/axios'

// In production, VITE_API_URL points to the Railway backend (e.g., https://your-app.railway.app)
// In development, empty string uses the Vite proxy (forwards /api to http://localhost:8080)
const API_BASE_URL = import.meta.env.VITE_API_URL || ''

// Debug logging helper
const logApiCall = (endpoint: string, data: any) => {
  console.log(`🌐 API ${endpoint}:`, {
    hasData: !!data,
    keys: data ? Object.keys(data) : [],
    sample: data ? JSON.stringify(data).substring(0, 200) : null
  })
}

// Training API
export const trainingApi = {
  /**
   * Start a new training session
   */
  async start(config: {
    domain_type: string
    grid_size: number
    n_episodes: number
    batch_size: number
    learning_rate: number
    objective: string
    hidden_dim?: number
    temperature?: number
    reward_peaks: Array<{ position: number[], intensity: number }>
    // Exploration parameters (Phase 7: Mode Collapse Fix)
    epsilon?: number
    epsilon_decay?: boolean
    entropy_weight?: number
    z_learning_rate_multiplier?: number
    // Experience Replay Buffer (JMLR 2023: Off-Policy Learning)
    use_replay_buffer?: boolean
    replay_buffer_size?: number
    replay_ratio?: number
    replay_priority_alpha?: number
    // TLM (ICLR 2025: Trajectory Likelihood Maximization)
    tlm_backward_weight?: number
    tlm_entropy_coeff?: number
    // Reward Shaping (auto-compensate path asymmetry)
    reward_shaping?: boolean
    // Domain-specific config (Phase 1: Domain Registry)
    domain_config?: Record<string, unknown>
  }) {
    const response = await axios.post(`${API_BASE_URL}/api/v2/training/start`, config)
    return response.data
  },

  /**
   * Get current training state (real-time metrics)
   */
  async getState() {
    const response = await axios.get(`${API_BASE_URL}/api/v2/training/state`)
    return response.data
  },

  /**
   * Get full training history
   */
  async getHistory() {
    const response = await axios.get(`${API_BASE_URL}/api/v2/training/history`)
    return response.data
  },

  /**
   * Stop training
   */
  async stop() {
    const response = await axios.post(`${API_BASE_URL}/api/v2/training/stop`)
    return response.data
  },

  /**
   * Pause training
   */
  async pause() {
    const response = await axios.post(`${API_BASE_URL}/api/v2/training/pause`)
    return response.data
  },

  /**
   * Extend training from current state (add more iterations without restarting)
   */
  async extend(additionalIterations: number) {
    const response = await axios.post(`${API_BASE_URL}/api/v2/training/extend`, {
      additional_iterations: additionalIterations,
    })
    return response.data
  },
}

// Trajectories API
export const trajectoriesApi = {
  /**
   * Get recent trajectories
   */
  async getRecent(limit?: number) {
    const params = limit ? { limit } : {}
    const response = await axios.get(`${API_BASE_URL}/api/v2/trajectories`, { params })
    const data = response.data

    // Debug logging
    logApiCall('/api/v2/trajectories', data)

    // Validate response structure
    if (!data) {
      console.error('❌ Trajectories API returned null/undefined')
      return { domain: null, trajectories: [], count: 0 }
    }

    if (!data.domain) {
      console.warn('⚠️ Trajectories API response missing domain field:', data)
    }

    if (!data.trajectories || !Array.isArray(data.trajectories)) {
      console.warn('⚠️ Trajectories API response missing trajectories array:', data)
    }

    return data
  },
}

// Analysis API
export const analysisApi = {
  /**
   * Get flow field data for visualization
   */
  async getFlowField() {
    const response = await axios.get(`${API_BASE_URL}/api/v2/analysis/flow`)
    return response.data
  },

  /**
   * Get distribution data (empirical vs target)
   */
  async getDistribution() {
    const response = await axios.get(`${API_BASE_URL}/api/v2/analysis/distribution`)
    return response.data
  },
}

// Domain API
export const domainApi = {
  /**
   * Get domain configuration and capabilities (legacy)
   */
  async getInfo() {
    const response = await axios.get(`${API_BASE_URL}/api/v2/domain/info`)
    return response.data
  },

  /**
   * List all registered domains (Phase 1: Domain Registry)
   */
  async list() {
    const response = await axios.get(`${API_BASE_URL}/api/v2/domains`)
    return response.data
  },

  /**
   * Get detailed info about a specific domain
   */
  async get(id: string) {
    const response = await axios.get(`${API_BASE_URL}/api/v2/domains/${id}`)
    return response.data
  },

  /**
   * Get JSON Schema for a domain's configuration
   */
  async getSchema(id: string) {
    const response = await axios.get(`${API_BASE_URL}/api/v2/domains/${id}/schema`)
    return response.data
  },

  /**
   * Validate configuration for a domain
   */
  async validate(id: string, config: Record<string, unknown>) {
    const response = await axios.post(`${API_BASE_URL}/api/v2/domains/${id}/validate`, config)
    return response.data
  },
}

// Molecular Generation API — connected to real Julia backend
export const molecularApi = {
  async generate(config: MolecularGenerationConfig) {
    const response = await axios.post(`${API_BASE_URL}/api/v2/molecular/generate`, config)
    return response.data
  },

  async getMolecules(params?: { limit?: number; offset?: number; sort_by?: string; filter?: Record<string, unknown> }) {
    const response = await axios.get(`${API_BASE_URL}/api/v2/molecular/molecules`, { params })
    return response.data
  },

  async getMolecule(id: string) {
    const response = await axios.get(`${API_BASE_URL}/api/v2/molecular/molecules/${id}`)
    return response.data
  },

  async compareMolecules(ids: string[]) {
    const response = await axios.post(`${API_BASE_URL}/api/v2/molecular/molecules/compare`, { ids })
    return response.data
  },

  async getChemicalSpace(params?: { method?: 'umap' | 'tsne' | 'pca'; color_by?: string }) {
    const response = await axios.get(`${API_BASE_URL}/api/v2/molecular/space`, { params })
    return response.data
  },

  async getAttribution(id: string) {
    const response = await axios.get(`${API_BASE_URL}/api/v2/molecular/attribution/${id}`)
    return response.data
  },

  async getRewardDecomposition(id: string) {
    const response = await axios.get(`${API_BASE_URL}/api/v2/molecular/reward-decomposition/${id}`)
    return response.data
  },

  async getGenerationDAG(id: string) {
    const response = await axios.get(`${API_BASE_URL}/api/v2/molecular/generation-dag/${id}`)
    return response.data
  },

  async retrain(config: { molecule_ids: string[]; additional_config?: Record<string, unknown> }) {
    const response = await axios.post(`${API_BASE_URL}/api/v2/molecular/retrain`, config)
    return response.data
  },

  async exportMolecules(params: { ids: string[]; format: 'smiles' | 'sdf' | 'csv' | 'png' }) {
    const response = await axios.post(`${API_BASE_URL}/api/v2/molecular/export`, params, {
      responseType: params.format === 'png' ? 'blob' : 'text',
    })
    return response.data
  },

  async getADMET(id: string) {
    const response = await axios.get(`${API_BASE_URL}/api/v2/molecular/admet/${id}`)
    return response.data
  },

  async validateSmiles(smiles: string) {
    const response = await axios.post(`${API_BASE_URL}/api/v2/molecular/validate-smiles`, { smiles })
    return response.data
  },
}

// Combined API object for convenience
export const api = {
  training: trainingApi,
  trajectories: trajectoriesApi,
  analysis: analysisApi,
  domain: domainApi,
  molecular: molecularApi,
}

// TypeScript types for API responses
export interface TrainingState {
  has_session: boolean
  current_iteration: number
  total_iterations: number
  is_training: boolean
  is_paused: boolean
  is_real_training?: boolean
  progress?: number
  // Latest scalar metrics (match backend field names)
  latest_loss?: number
  latest_reward?: number
  latest_gradient_norm?: number
  last_error: string | null
  error_count?: number
  // Exploration metrics
  current_epsilon?: number
  epsilon_decay?: boolean
  entropy_weight?: number
  z_learning_rate_multiplier?: number
  // Computed aggregate metrics
  metrics?: {
    mean_reward?: number
    diversity_ratio?: number
    mean_log_Z?: number
    mean_trajectory_length?: number
  }
  domain_metrics?: Record<string, unknown>
  // Molecular domain stats (returned inline when domain is molecule)
  total_molecules?: number
  unique_smiles?: number
  // Flow statistics
  flow_statistics?: {
    mean_log_Z?: number
    mean_reward?: number
    progress?: number
    mean_trajectory_length?: number
  }
}

export interface TrainingHistory {
  losses: number[]
  rewards: number[]
  gradient_norms: number[]
  iteration_times: number[]
}

export interface Trajectory {
  id: string
  states: any[]
  actions: any[]
  rewards: number[]
  total_reward: number
  length: number
}

export interface FlowField {
  supported: boolean
  grid_size?: number
  data?: Array<{
    position: number[]
    velocity: number[]
    magnitude: number
    flow: number
  }>
}

export interface Distribution {
  supported: boolean
  grid_size?: number
  empirical?: number[][]
  target?: number[][]
  counts?: number[][]
  total_samples?: number
}

export interface DomainInfo {
  domain_type: string
  renderer: string
  capabilities: {
    supports_flow_field: boolean
    supports_distribution: boolean
    supports_domain_metrics: boolean
  }
  config: any
}

// Molecular Generation Types
export interface MolecularGenerationConfig {
  method: 'de_novo' | 'scaffold_hopping' | 'fragment_linking' | 'r_group' | 'optimization' | 'grid_world'
  seed_smiles?: string
  scaffold_smiles?: string
  constraints?: MolecularConstraints
  gflownet_config?: {
    objective: string
    n_episodes: number
    batch_size: number
    learning_rate: number
    hidden_dim?: number
    temperature?: number
    epsilon?: number
    entropy_weight?: number
  }
  preset?: string
}

export interface MolecularConstraints {
  mw_range?: [number, number]
  logp_range?: [number, number]
  qed_range?: [number, number]
  sa_range?: [number, number]
  tpsa_range?: [number, number]
  rotatable_bonds_range?: [number, number]
  hbd_range?: [number, number]
  hba_range?: [number, number]
  lipinski?: boolean
  veber?: boolean
  pains_filter?: boolean
  brenk_filter?: boolean
}

export interface Molecule {
  id: string
  smiles: string
  properties: MolecularProperties
  reward: number
  generation_step: number
  method: string
  svg_2d?: string
  fingerprint?: number[]
  created_at?: string
  coords_3d?: Array<{ atom: string; x: number; y: number; z: number }>
  bonds?: Array<{ from: number; to: number; order: number }>
}

export interface MolecularProperties {
  molecular_weight: number
  logp: number
  qed: number
  synthetic_accessibility: number
  tpsa: number
  rotatable_bonds: number
  hbd: number
  hba: number
  num_rings: number
  num_aromatic_rings: number
  formula?: string
}

export interface ChemicalSpacePoint {
  id: string
  x: number
  y: number
  smiles: string
  reward: number
  properties: MolecularProperties
  cluster_id?: number
  generation_epoch?: number
}

export interface AtomAttribution {
  molecule_id: string
  smiles: string
  atom_scores: number[]
  bond_scores?: number[]
  attribution_type: 'reward' | 'flow' | 'loss'
}

export interface RewardDecompositionData {
  molecule_id: string
  components: Array<{
    name: string
    value: number
    weight: number
    contribution: number
  }>
  total_reward: number
}

export default api
