# Convergence assertions shared by the example scripts.
#
# WHY THIS FILE EXISTS
# --------------------
# `train_gflownet` wraps each iteration in a try/catch and, on failure, records a
# NaN loss and keeps going (src/training/training.jl, the `catch e` at the bottom of
# the iteration loop). A training run can therefore "complete" having learned
# absolutely nothing: every single iteration raised, `history.losses` is all NaN,
# and the calling script prints its summary and exits 0.
#
# That has bitten this repo repeatedly. `sub_trajectory_balance_demo.jl` was
# announcing "=== Demo Complete ===" from a run in which every STB iteration threw
# `ArgumentError: SUB_TRAJECTORY_BALANCE requires a flow estimator`.
#
# So: every example that trains MUST call `assert_finite_iterations`. It needs no
# tuned threshold and no measured constant — the number of finite losses is simply
# required to equal the number of iterations requested. A silently-failing run
# cannot get past it.
#
# The other helpers are quality checks with thresholds. Each call site states the
# value it was measured against.

using Statistics: mean

"""
    assert_finite_iterations(history, n_expected, label) -> Int

Require that the run recorded exactly `n_expected` losses and that every one of
them is finite. This is the anti-silent-failure gate: `train_gflownet` records NaN
for any iteration that raised, so a shortfall here means the run did not train.

Raises on failure; returns the finite count on success.
"""
function assert_finite_iterations(history, n_expected::Integer, label::AbstractString)
    losses = history.losses
    if length(losses) != n_expected
        error("CONVERGENCE ASSERTION FAILED [$label]: recorded $(length(losses)) " *
              "losses but $n_expected iterations were requested.")
    end
    bad = findall(!isfinite, losses)
    if !isempty(bad)
        shown = length(bad) > 10 ? "$(bad[1:10]) (… $(length(bad)) total)" : "$bad"
        error("CONVERGENCE ASSERTION FAILED [$label]: only " *
              "$(n_expected - length(bad))/$n_expected iterations produced a finite " *
              "loss. train_gflownet swallows per-iteration exceptions and records " *
              "NaN, so this run learned nothing on iterations $shown. Re-run with " *
              "`verbose=true` to print the underlying exception.")
    end
    println("   ✓ [$label] all $n_expected iterations produced a finite loss")
    return n_expected
end

"""
    assert_loss_decreased(history, label; window, max_ratio) -> Float64

Require `mean(last window losses) <= max_ratio * mean(first window losses)`.

Only valid for objectives whose loss is positive (TB, TLM, FM, STB). If the opening
mean is not positive the ratio is meaningless, so this raises rather than passing
vacuously — use [`assert_final_loss_below`](@ref) for signed losses such as
DETAILED_BALANCE.

Returns the observed ratio.
"""
function assert_loss_decreased(history, label::AbstractString;
                               window::Integer = 10, max_ratio::Real = 0.5)
    losses = history.losses
    w = min(Int(window), length(losses) ÷ 2)
    w >= 1 || error("CONVERGENCE ASSERTION FAILED [$label]: only " *
                    "$(length(losses)) losses recorded, too few to compare a " *
                    "$(window)-iteration window against the start.")
    first_mean = mean(@view losses[1:w])
    last_mean = mean(@view losses[end - w + 1:end])
    if !(first_mean > 0)
        error("CONVERGENCE ASSERTION FAILED [$label]: opening loss mean is " *
              "$(first_mean), not positive, so a decrease RATIO is meaningless. " *
              "Use assert_final_loss_below for this objective.")
    end
    ratio = last_mean / first_mean
    if !(last_mean <= max_ratio * first_mean)
        error("CONVERGENCE ASSERTION FAILED [$label]: loss did not fall. " *
              "mean(first $w) = $(round(first_mean, digits=4)), " *
              "mean(last $w) = $(round(last_mean, digits=4)), " *
              "ratio = $(round(ratio, digits=4)), required <= $max_ratio.")
    end
    println("   ✓ [$label] loss fell $(round(first_mean, digits=4)) → " *
            "$(round(last_mean, digits=4)) (ratio $(round(ratio, digits=4)) <= $max_ratio)")
    return ratio
end

"""
    assert_loss_below_initial(history, label; window, max_ratio) -> Float64

Require `mean(last window losses) <= max_ratio * losses[1]`.

Prefer this over [`assert_loss_decreased`](@ref) whenever the loss collapses inside
the first one or two iterations, which makes an opening WINDOW mean already
post-descent and therefore a weak reference. `losses[1]` is the loss of the
essentially untrained network, which is the honest baseline.

It is also the statistic that discriminates a frozen run. Measured on
SUB_TRAJECTORY_BALANCE, 100 iterations, 6x6 grid, three seeds:
  converged: losses[1] = 14.41, mean(last 25) = 0.186  -> ratio 0.013
  frozen:    losses[1] = 185.80, mean(last 25) = 235.25 -> ratio 1.27
  frozen:    losses[1] = 68.96, mean(last 25) = 227.24  -> ratio 3.29

Returns the observed ratio.
"""
function assert_loss_below_initial(history, label::AbstractString;
                                   window::Integer = 10, max_ratio::Real = 0.25)
    losses = history.losses
    isempty(losses) && error("CONVERGENCE ASSERTION FAILED [$label]: no losses recorded.")
    w = min(Int(window), length(losses))
    initial = losses[1]
    if !(initial > 0)
        error("CONVERGENCE ASSERTION FAILED [$label]: initial loss is $(initial), " *
              "not positive, so a ratio against it is meaningless.")
    end
    last_mean = mean(@view losses[end - w + 1:end])
    ratio = last_mean / initial
    if !(last_mean <= max_ratio * initial)
        error("CONVERGENCE ASSERTION FAILED [$label]: loss did not fall below its " *
              "starting value. losses[1] = $(round(initial, digits=4)), " *
              "mean(last $w) = $(round(last_mean, digits=4)), " *
              "ratio = $(round(ratio, digits=4)), required <= $max_ratio.")
    end
    println("   ✓ [$label] loss fell $(round(initial, digits=4)) → " *
            "$(round(last_mean, digits=4)) over the run " *
            "(ratio $(round(ratio, digits=4)) <= $max_ratio)")
    return ratio
end

"""
    assert_final_loss_below(history, label; threshold, window) -> Float64

Require `mean(last window losses) <= threshold`. Use for objectives whose loss can
go negative (DETAILED_BALANCE), where a decrease ratio does not typecheck as a
notion of progress. `threshold` must come from a measurement recorded at the call
site.

Returns the observed final mean.
"""
function assert_final_loss_below(history, label::AbstractString;
                                 threshold::Real, window::Integer = 10)
    losses = history.losses
    w = min(Int(window), length(losses))
    w >= 1 || error("CONVERGENCE ASSERTION FAILED [$label]: no losses recorded.")
    last_mean = mean(@view losses[end - w + 1:end])
    if !(last_mean <= threshold)
        error("CONVERGENCE ASSERTION FAILED [$label]: mean loss over the final " *
              "$w iterations is $(round(last_mean, digits=4)), required <= $threshold.")
    end
    println("   ✓ [$label] final loss mean $(round(last_mean, digits=4)) <= $threshold")
    return last_mean
end

"""
    assert_modes_discovered(counts, label; min_per_mode, n_samples) -> Int

Require that at least `min_per_mode` of the drawn samples landed on EVERY mode in
`counts`. Use where the script's scientific claim is coverage rather than loss.

Returns the number of modes that cleared the bar (always `length(counts)` on
success).
"""
function assert_modes_discovered(counts, label::AbstractString;
                                 min_per_mode::Integer, n_samples::Integer)
    missed = [i for (i, c) in enumerate(counts) if c < min_per_mode]
    if !isempty(missed)
        error("CONVERGENCE ASSERTION FAILED [$label]: modes $missed drew fewer " *
              "than $min_per_mode of $n_samples samples (counts = $(collect(counts))). " *
              "The trained sampler is not covering every reward mode.")
    end
    println("   ✓ [$label] every mode drew >= $min_per_mode/$n_samples samples " *
            "(counts = $(collect(counts)))")
    return length(counts)
end

"""
    assert_beats_untrained(trained_rewards, untrained_rewards, label; min_gain) -> Float64

Require that the trained sampler's mean terminal reward be at least `min_gain` times
an untrained model's mean terminal reward.

Use this INSTEAD of a loss check for any run configured with
`SIMPLE_ESTIMATION`. That method pins log Z = 0, i.e. Z = 1
(src/training/losses.jl: "SIMPLE_ESTIMATION: Z = 1, so log Z = 0"), so the
trajectory-balance residual `(log P(tau) - log R)^2` cannot be driven to zero
whenever R > 1 -- the loss is bounded away from 0 and MEASURABLY RISES as the
policy concentrates. A decrease assertion on such a run would be asserting
something false. What training does deliver there is a sampler that concentrates on
high-reward terminals, and that is what this checks.

Returns the observed gain factor.
"""
function assert_beats_untrained(trained_rewards, untrained_rewards,
                                label::AbstractString; min_gain::Real = 1.2)
    isempty(trained_rewards) && error("CONVERGENCE ASSERTION FAILED [$label]: no trained samples.")
    isempty(untrained_rewards) && error("CONVERGENCE ASSERTION FAILED [$label]: no baseline samples.")
    t = mean(trained_rewards)
    u = mean(untrained_rewards)
    if !(u > 0)
        error("CONVERGENCE ASSERTION FAILED [$label]: untrained baseline mean " *
              "reward is $(u), not positive, so a gain factor is meaningless.")
    end
    gain = t / u
    if !(gain >= min_gain)
        error("CONVERGENCE ASSERTION FAILED [$label]: trained mean reward " *
              "$(round(t, digits=3)) vs untrained $(round(u, digits=3)) = " *
              "$(round(gain, digits=3))x, required >= $(min_gain)x. Training did " *
              "not move the sampler toward high-reward terminals.")
    end
    println("   ✓ [$label] mean reward $(round(u, digits=3)) (untrained) → " *
            "$(round(t, digits=3)) (trained) = $(round(gain, digits=3))x >= $(min_gain)x")
    return gain
end

"""
    assert_relative_error_below(observed, expected, label; max_rel_error) -> Float64

Require `|observed - expected| / |expected| <= max_rel_error`. Use where the demo
has an exact analytic target (e.g. the 2×2 grid partition function Z = 4R).
"""
function assert_relative_error_below(observed::Real, expected::Real,
                                     label::AbstractString; max_rel_error::Real)
    isfinite(observed) || error("CONVERGENCE ASSERTION FAILED [$label]: observed " *
                                "value is $observed, not finite.")
    expected != 0 || error("CONVERGENCE ASSERTION FAILED [$label]: expected value " *
                           "is zero; relative error is undefined.")
    rel = abs(observed - expected) / abs(expected)
    if !(rel <= max_rel_error)
        error("CONVERGENCE ASSERTION FAILED [$label]: observed $(observed), " *
              "expected $(expected), relative error $(round(rel * 100, digits=2))%, " *
              "required <= $(max_rel_error * 100)%.")
    end
    println("   ✓ [$label] $(round(observed, digits=4)) vs target " *
            "$(round(expected, digits=4)) — relative error " *
            "$(round(rel * 100, digits=2))% <= $(max_rel_error * 100)%")
    return rel
end

"""
    warn_loss_below_initial(history, label; window, max_ratio, reason) -> Bool

Same test as [`assert_loss_below_initial`](@ref), but reports a loud `@warn` with
`reason` instead of raising.

Use this ONLY where the check is currently UNSATISFIABLE because of a defect
outside the example's control, and record why in `reason`. It is not a way to make a
script green: the check still runs, still prints its numbers, and still says plainly
that training failed. When the underlying defect is fixed this call should be
promoted back to `assert_loss_below_initial`.

Returns whether the check passed.
"""
function warn_loss_below_initial(history, label::AbstractString;
                                 window::Integer = 10, max_ratio::Real = 0.25,
                                 reason::AbstractString)
    losses = history.losses
    isempty(losses) && error("CONVERGENCE CHECK [$label]: no losses recorded.")
    w = min(Int(window), length(losses))
    initial = losses[1]
    last_mean = mean(@view losses[end - w + 1:end])
    passed = initial > 0 && last_mean <= max_ratio * initial
    if passed
        println("   ✓ [$label] loss fell $(round(initial, digits=4)) → " *
                "$(round(last_mean, digits=4)) " *
                "(ratio $(round(last_mean / initial, digits=4)) <= $max_ratio)")
    else
        @warn """
              CONVERGENCE CHECK NOT MET [$label] -- reported, NOT asserted.
                losses[1]      = $(round(initial, digits=4))
                mean(last $w)  = $(round(last_mean, digits=4))
                ratio          = $(round(last_mean / initial, digits=4)) (wanted <= $max_ratio)
              $reason
              """
    end
    return passed
end
