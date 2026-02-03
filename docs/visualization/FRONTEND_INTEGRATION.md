# Frontend Integration - Real Training Visualization

**Status**: 🚧 In Progress
**Date**: February 2, 2026
**Backend**: ✅ Complete (v2 API)
**Frontend**: 🚧 Partially integrated

---

## Overview

Connecting the existing React/TypeScript/Three.js frontend to the real GFlowNet training backend (unified_server.jl) to show live training visualization.

---

## ✅ Completed

### 1. API Service Layer Created
**File**: `src/utils/visualization/web/src/services/api.ts`

Created centralized API service with TypeScript types:
- `trainingApi`: start, getState, getHistory, stop, pause
- `trajectoriesApi`: getRecent
- `analysisApi`: getFlowField, getDistribution
- `domainApi`: getInfo

All methods connect to v2 endpoints (`/api/v2/*`).

### 2. Updated Components

#### RealtimeMetrics.tsx ✅
- ✅ Imports new API service
- ✅ Uses `api.training.getState()` instead of old `/api/training/metrics`
- ✅ Uses `api.trajectories.getRecent()` instead of `/api/trajectories`
- ✅ Updated metrics display to use v2 API response fields:
  - `loss`, `mean_reward`, `gradient_norm`
  - `current_iteration`, `total_iterations`
  - `is_training` status
- ✅ Faster polling: 250ms for real-time feel

#### App.tsx ✅
- ✅ Imports API service
- ✅ Uses `api.training.start()` instead of old `/api/training/reset`
- ✅ Proper error handling with user feedback
- ✅ Maps frontend config to backend v2 API format

---

## 🚧 Todo: Remaining Components

### High Priority

#### 1. GFlowNetTrainingDashboard.tsx
**What it does**: Shows training history charts (loss, reward over time)

**Needs updating**:
```typescript
// Current: Mock data or old API
const { data: history } = useQuery(['training-history'], ...)

// Update to:
const { data: history } = useQuery({
  queryKey: ['training-history'],
  queryFn: async () => await api.training.getHistory(),
  refetchInterval: 1000
})

// Use history.losses, history.rewards, history.gradient_norms
```

#### 2. GFlowNet2DTrajectory.tsx
**What it does**: Shows recent trajectories in 2D

**Needs updating**:
```typescript
// Update to:
const { data: trajectories } = useQuery({
  queryKey: ['trajectories'],
  queryFn: async () => await api.trajectories.getRecent(10),
  refetchInterval: 500
})

// Use trajectories.trajectories array
```

#### 3. GFlowNetDistribution3D.tsx
**What it does**: 3D visualization of state distribution

**Needs updating**:
```typescript
// Update to:
const { data: distribution } = useQuery({
  queryKey: ['distribution'],
  queryFn: async () => await api.analysis.getDistribution(),
  refetchInterval: 2000
})

// Use distribution.empirical, distribution.target
// distribution.grid_size for dimensions
```

#### 4. GFlowNetFlowField.tsx
**What it does**: Shows policy flow field

**Needs updating**:
```typescript
// Update to:
const { data: flowField } = useQuery({
  queryKey: ['flow-field'],
  queryFn: async () => await api.analysis.getFlowField(),
  refetchInterval: 2000
})

// Use flowField.data array
// Each point has: position, velocity, magnitude, flow
```

### Medium Priority

#### 5. ProblemSetup.tsx
Verify that config format matches what App.tsx expects and sends to backend.

#### 6. DomainInfo.tsx
Update to fetch domain capabilities:
```typescript
const { data: domainInfo } = useQuery({
  queryKey: ['domain-info'],
  queryFn: async () => await api.domain.getInfo()
})
```

---

## 🧪 Testing Steps

### 1. Start Backend Server
```bash
cd /path/to/GFlowNet.jl
julia --project=. -e '
include("src/utils/visualization/api/unified_server.jl")
start_real_training_server(port=8080)
'
```

### 2. Start Frontend Dev Server
```bash
cd src/utils/visualization/web
npm install  # if needed
npm run dev
```

### 3. Test Flow
1. Open `http://localhost:5173` (or whatever Vite shows)
2. Go to Setup tab
3. Configure grid world problem
4. Click "Start Training"
5. Should see real training in Monitor tab:
   - Live metrics updating (loss, reward)
   - Training progress bar filling
   - Real gradient descent happening

### 4. Verify API Calls
Open browser DevTools → Network tab:
- Should see calls to `http://localhost:8080/api/v2/training/state` every 250ms
- Should see `POST` to `/api/v2/training/start` when starting
- Should see data flowing with real numbers (not mocks)

---

## 🐛 Known Issues & Fixes

### Issue 1: CORS Errors
**Symptom**: Console shows "CORS policy" errors

**Fix**: Backend server already has CORS enabled in unified_server.jl:
```julia
@get "/" function()
    headers = Dict("Access-Control-Allow-Origin" => "*")
    return Dict("message" => "GFlowNet Training API v2")
end
```

If still seeing errors, check browser console for exact URL being called.

### Issue 2: Connection Refused
**Symptom**: "Failed to fetch" or "Network error"

**Fix**:
1. Verify backend is running: `curl http://localhost:8080/api/v2/domain/info`
2. Check port 8080 is not blocked by firewall
3. Verify `API_BASE_URL` in `api.ts` matches server port

### Issue 3: TypeScript Errors
**Symptom**: Build errors about missing types

**Fix**: Types are defined in `api.ts`. If component needs more specific types, add them:
```typescript
import type { TrainingState, TrainingHistory } from '../services/api'
```

---

## 📊 API Response Examples

### Training State
```json
{
  "status": "ok",
  "current_iteration": 42,
  "total_iterations": 200,
  "is_training": true,
  "is_paused": false,
  "loss": 15.42,
  "mean_reward": 3.75,
  "gradient_norm": 8.91,
  "learning_rate": 0.01,
  "last_error": null
}
```

### Training History
```json
{
  "losses": [25.3, 18.2, 15.4, ...],
  "rewards": [2.1, 2.8, 3.5, ...],
  "gradient_norms": [12.4, 9.8, 8.1, ...],
  "iterations": [1, 2, 3, ...]
}
```

### Trajectories
```json
{
  "trajectories": [
    {
      "id": "traj-1",
      "states": [...],
      "actions": [...],
      "rewards": [0, 0, 10.0],
      "total_reward": 10.0,
      "length": 3
    }
  ]
}
```

### Flow Field
```json
{
  "supported": true,
  "grid_size": 8,
  "data": [
    {
      "position": [1, 1],
      "velocity": [0.8, 0.2],
      "magnitude": 0.82,
      "flow": 5.2
    },
    ...
  ]
}
```

### Distribution
```json
{
  "supported": true,
  "grid_size": 8,
  "empirical": [[0.1, 0.05, ...], ...],  // 8×8 matrix
  "target": [[0.08, 0.04, ...], ...],     // 8×8 matrix
  "counts": [[10, 5, ...], ...],          // Visit counts
  "total_samples": 100
}
```

---

## 🎯 Next Steps

1. **Update remaining components** (listed above in Todo section)
2. **Test end-to-end** with real backend running
3. **Fix any bugs** that appear during testing
4. **Polish UI** - add loading states, error messages, etc.
5. **Add features**:
   - Pause/Resume buttons
   - Stop training button
   - Export data button
   - Settings panel

---

## 📚 File Structure

```
src/utils/visualization/web/src/
├── services/
│   └── api.ts                    ✅ NEW - Centralized API service
├── components/
│   ├── App.tsx                   ✅ Updated
│   ├── RealtimeMetrics.tsx       ✅ Updated
│   ├── GFlowNetTrainingDashboard.tsx  🚧 TODO
│   ├── MonitoringDashboard.tsx   ✅ Uses updated components
│   ├── ProblemSetup.tsx          🔍 Verify
│   └── DomainInfo.tsx            🚧 TODO
└── visualizations/
    ├── GFlowNet2DTrajectory.tsx  🚧 TODO
    ├── GFlowNetDistribution3D.tsx 🚧 TODO
    └── GFlowNetFlowField.tsx     🚧 TODO
```

---

## 🚀 Estimated Time to Complete

- Remaining component updates: **30-60 minutes**
- Testing and debugging: **30-60 minutes**
- Polish and refinement: **30 minutes**

**Total**: ~2-3 hours to fully functional real-time visualization

---

**Updated**: February 2, 2026
**Status**: 40% complete (API service + 2 components done)
