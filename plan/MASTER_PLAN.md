# 🚀 GFlowNet Package Modernization Plan

## 📋 Overview

This plan addresses the critical inconsistencies between the modern core framework and outdated examples. The goal is to modernize all examples to use the new training interface and ensure full integration with the core package.

## 🎯 Objectives

1. **Modernize Examples**: Update all examples to use `TrainingConfig` + `train_gflownet()` pattern
2. **Fix Interface Inconsistencies**: Ensure examples use core DAG, policies, and training infrastructure  
3. **Integrate Applications**: Connect examples with `src/applications/` modules
4. **Fix Critical Bugs**: Address zero rewards, gradient computation issues, and other blockers

## 📊 Priority Matrix

### 🔴 **CRITICAL (Must Fix)**
- [ ] **P1-GRID-WORLD**: Fix grid world to use modern interface
- [ ] **P1-REWARDS**: Fix reward functions (zero rewards break GFlowNet)
- [ ] **P1-INTEGRATION**: Connect examples with applications module

### 🟡 **HIGH (Should Fix)**  
- [ ] **P2-FEATURE-ACQ**: Modernize feature acquisition example
- [ ] **P2-DAG-USAGE**: Use core DAG instead of custom implementations
- [ ] **P2-VALIDATION**: Add integration tests

### 🟢 **MEDIUM (Nice to Have)**
- [ ] **P3-DOCS**: Update documentation for new interfaces
- [ ] **P3-MIGRATION**: Create migration guide
- [ ] **P3-POLISH**: Add more utilities and visualizations

## 📁 Implementation Structure

```
plan/
├── MASTER_PLAN.md                    # This file
├── PHASE_1_CRITICAL_FIXES.md         # Critical priority items
├── PHASE_2_MODERNIZATION.md          # High priority modernization
├── PHASE_3_POLISH.md                 # Nice-to-have improvements
└── IMPLEMENTATION_LOG.md             # Progress tracking
```

## 🎯 Success Criteria

### Phase 1 (Critical)
- ✅ Grid world example runs with modern training interface
- ✅ All reward functions return positive values
- ✅ Examples integrate with applications module
- ✅ No more custom training loops in examples

### Phase 2 (High Priority)
- ✅ Feature acquisition uses core DAG structure
- ✅ All examples use `TrainingConfig` pattern
- ✅ Integration tests pass
- ✅ Consistent state/action definitions

### Phase 3 (Polish)
- ✅ Comprehensive documentation
- ✅ Migration guide for old → new interface
- ✅ Enhanced visualization utilities
- ✅ Performance benchmarks

## 📅 Timeline

- **Phase 1**: 1-2 hours (Critical fixes)
- **Phase 2**: 2-3 hours (Modernization)  
- **Phase 3**: 1-2 hours (Polish)
- **Total**: 4-7 hours

## 🔧 Implementation Approach

1. **Start with Phase 1**: Fix critical blockers first
2. **Test Each Fix**: Verify each change works before moving on
3. **Document Progress**: Update implementation log after each task
4. **Validate Integration**: Ensure examples work with core package

## 📝 Next Steps

1. Review and approve this master plan
2. Begin Phase 1 critical fixes
3. Track progress in implementation log
4. Move to subsequent phases after validation

---

*Created: $(date)* 
*Status: PLANNING* 