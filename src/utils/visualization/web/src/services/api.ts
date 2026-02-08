// API Service for GFlowNet Real Training Visualization
// Connects to the v2 backend endpoints implemented in unified_server.jl

import axios from '../lib/axios'

// Use empty string to use Vite proxy (recommended for development)
// The Vite proxy in vite.config.ts forwards /api to http://localhost:8080
// This avoids CORS issues in the browser
const API_BASE_URL = ''

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

// Combined API object for convenience
export const api = {
  training: trainingApi,
  trajectories: trajectoriesApi,
  analysis: analysisApi,
  domain: domainApi,
}

// TypeScript types for API responses
export interface TrainingState {
  status: string
  current_iteration: number
  total_iterations: number
  is_training: boolean
  is_paused: boolean
  loss: number
  latest_loss?: number
  mean_reward: number
  gradient_norm: number
  latest_gradient_norm?: number
  learning_rate: number
  last_error: string | null
  error_count?: number
  // Exploration metrics (Phase 7: Mode Collapse Fix)
  current_epsilon?: number
  epsilon_decay?: boolean
  metrics?: {
    mean_reward?: number
    diversity_ratio?: number
  }
}

export interface TrainingHistory {
  losses: number[]
  rewards: number[]
  gradient_norms: number[]
  iterations: number[]
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

export default api
