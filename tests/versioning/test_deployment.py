import pytest
from drawbore.errors import DrawboreError
from drawbore.versioning import Deployment, RolloutPlan, DeploymentError


def test_deployment_error_is_a_drawbore_error():
    assert issubclass(DeploymentError, DrawboreError)


def test_begin_rejects_a_second_rollout_while_one_is_in_flight():
    dep = Deployment(stable_version="1.0.0")
    dep.begin(RolloutPlan(change_kind="non_breaking", new_version="1.1.0"))
    # a rollout is already in flight; must rollback() before beginning another
    with pytest.raises(DeploymentError):
        dep.begin(RolloutPlan(change_kind="breaking", new_version="2.0.0"))
    # after an explicit rollback, a new rollout may begin
    dep.rollback()
    dep.begin(RolloutPlan(change_kind="breaking", new_version="2.0.0"))
    assert dep.in_flight is not None


def test_breaking_change_must_start_in_shadow():
    plan = RolloutPlan(change_kind="breaking", new_version="2.0.0")
    assert plan.stages == ("shadow", "canary", "full")
    assert plan.stage == "shadow"


def test_non_breaking_change_skips_shadow():
    plan = RolloutPlan(change_kind="non_breaking", new_version="1.1.0")
    assert plan.stages == ("canary", "full")
    assert plan.stage == "canary"


def test_cannot_advance_a_stage_that_has_not_passed():
    plan = RolloutPlan(change_kind="non_breaking", new_version="1.1.0")
    with pytest.raises(DeploymentError):
        plan.advance()  # canary not marked passed


def test_advance_walks_stages_in_order_to_completion():
    plan = RolloutPlan(change_kind="breaking", new_version="2.0.0")
    plan.mark_passed()           # shadow passed
    plan.advance()
    assert plan.stage == "canary"
    assert plan.is_complete is False
    plan.mark_passed()           # canary passed
    plan.advance()
    assert plan.stage == "full"
    assert plan.is_complete is False
    plan.mark_passed()           # full passed
    assert plan.is_complete is True
    with pytest.raises(DeploymentError):
        plan.advance()           # already at final stage


def test_complete_promotes_new_version_to_stable():
    dep = Deployment(stable_version="1.0.0")
    plan = RolloutPlan(change_kind="non_breaking", new_version="1.1.0")
    dep.begin(plan)
    plan.mark_passed(); plan.advance()   # canary -> full
    plan.mark_passed()                   # full passed -> complete
    dep.complete()
    assert dep.stable_version == "1.1.0"


def test_cannot_complete_an_incomplete_rollout():
    dep = Deployment(stable_version="1.0.0")
    plan = RolloutPlan(change_kind="breaking", new_version="2.0.0")
    dep.begin(plan)
    plan.mark_passed()  # only shadow passed
    with pytest.raises(DeploymentError):
        dep.complete()


def test_rollback_returns_to_last_stable_and_discards_the_rollout():
    dep = Deployment(stable_version="1.0.0")
    plan = RolloutPlan(change_kind="breaking", new_version="2.0.0")
    dep.begin(plan)
    plan.mark_passed(); plan.advance()   # in canary now
    assert dep.rollback() == "1.0.0"     # a failed gate routes back to stable
    assert dep.stable_version == "1.0.0"
    assert dep.in_flight is None         # rollout discarded


def test_unknown_change_kind_is_rejected_at_construction():
    with pytest.raises(DeploymentError):
        RolloutPlan(change_kind="BREAKING", new_version="9.9.9")  # not a valid ChangeKind


def test_begin_requires_a_fresh_plan():
    dep = Deployment(stable_version="1.0.0")
    plan = RolloutPlan(change_kind="non_breaking", new_version="1.1.0")
    plan.mark_passed()
    plan.advance()  # plan is no longer fresh (advanced to its final stage)
    with pytest.raises(DeploymentError):
        dep.begin(plan)


def test_complete_raises_when_no_rollout_in_flight():
    dep = Deployment(stable_version="1.0.0")
    with pytest.raises(DeploymentError):
        dep.complete()


def test_complete_promotes_breaking_new_version_to_stable():
    dep = Deployment(stable_version="1.0.0")
    plan = RolloutPlan(change_kind="breaking", new_version="2.0.0")
    dep.begin(plan)
    plan.mark_passed(); plan.advance()  # shadow -> canary
    plan.mark_passed(); plan.advance()  # canary -> full
    plan.mark_passed()                  # full passed
    dep.complete()
    assert dep.stable_version == "2.0.0"
    assert dep.in_flight is None
