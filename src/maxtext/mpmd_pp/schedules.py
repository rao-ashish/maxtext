"""Implementations of various schedules for MPMD pipeline parallelism.

Currently supported: GPipe, 1F1B (both with circular repeat).
"""

from enum import Enum
from queue import SimpleQueue
from collections import defaultdict


# ==== Section, Task, and Schedule abstractions ==== #


class SectionKind(Enum):
    STAGE_INIT = 1
    FORWARD = 2
    BACKWARD = 3
    FUSED_BACKWARD_UPDATE = 4
    UPDATE = 5


# (section_kind, logical_stage_idx)
SectionName = tuple[SectionKind, int]

# (microbatch_idx, section_name). microbatch_idx is None for sections like
# STAGE_INIT and UPDATE that are not specific to a microbatch.
Task = tuple[int | None, SectionName]
PipelineSchedule = list[Task]


def task_to_str(task):
    microbatch_idx, (section_kind, logical_stage_idx) = task
    return f"Task(mu={microbatch_idx}|s={logical_stage_idx}|k={section_kind})"


# ==== Schedule generators ==== #


# ---- GPipe schedule ---- #


def make_gpipe_schedule(num_microbatches, num_logical_stages) -> PipelineSchedule:
    # Init tasks.
    init_tasks = [
        (None, (SectionKind.STAGE_INIT, logical_stage_idx))
        for logical_stage_idx in range(num_logical_stages)
    ]

    # Forward tasks. We sort by wavefront schedule to ensure good comm/compute
    # overlap.
    forward_tasks = [
        (microbatch_idx, (SectionKind.FORWARD, logical_stage_idx))
        for microbatch_idx in range(num_microbatches)
        for logical_stage_idx in range(num_logical_stages)
    ]

    def forward_task_key(task):
        microbatch_idx, (_, logical_stage_idx) = task
        return (microbatch_idx + logical_stage_idx, logical_stage_idx)

    forward_tasks.sort(key=forward_task_key)

    # Backward tasks. For the last microbatch of stage 0, we do a fused
    # bwd/update. We sort by wavefront schedule to ensure good comm/compute
    # overlap.
    def get_backward_task(microbatch_idx, logical_stage_idx):
        if microbatch_idx == num_microbatches - 1 and logical_stage_idx == 0:
            return (
                microbatch_idx,
                (SectionKind.FUSED_BACKWARD_UPDATE, logical_stage_idx),
            )
        return (microbatch_idx, (SectionKind.BACKWARD, logical_stage_idx))

    backward_tasks = [
        get_backward_task(microbatch_idx, logical_stage_idx)
        for microbatch_idx in range(num_microbatches)
        for logical_stage_idx in range(num_logical_stages)
    ]

    def backward_task_key(task):
        microbatch_idx, (_, logical_stage_idx) = task
        return (microbatch_idx - logical_stage_idx, -1 * logical_stage_idx)

    backward_tasks.sort(key=backward_task_key)

    # Update tasks. We start at stage 1 because stage 0 did an update as part
    # of its last backward pass.
    update_tasks = [
        (None, (SectionKind.UPDATE, logical_stage_idx))
        for logical_stage_idx in range(1, num_logical_stages)
    ]

    return init_tasks + forward_tasks + backward_tasks + update_tasks


# ---- 1F1B schedule ---- #

# The implementations below are lightly modified versions of JaxPP schedules,
# and are taken from here:
# https://github.com/NVIDIA/jaxpp/blob/5fde1d2bc2e4ecaeea05dce08b931c5a790b9619/src/jaxpp/schedules.py


def serialize_aligned_jaxpp_schedule(aligned_schedule) -> PipelineSchedule:
    """Given a partitioned schedule that is aligned (all physical stage have
    schedules of the same length, and timesteps where a stage is idle is marked
    by a "None" task), serialize it into a flattened schedule."""

    # Verify that we were given an aligned schedule.
    assert all(
        len(schedule) == len(aligned_schedule[0]) for schedule in aligned_schedule
    )

    # Serialize the schedule.
    flat_schedule = []

    for timestep in range(len(aligned_schedule[0])):
        for physical_idx in range(len(aligned_schedule)):
            task = aligned_schedule[physical_idx][timestep]
            if task is not None:
                flat_schedule.append(task)

    return flat_schedule


# Helper function for serialize_unaligned_jaxpp_schedule and
# align_jaxpp_schedule.
def task_is_ready(task, done_tasks, num_logical_stages):
    microbatch_idx, (section_kind, logical_idx) = task
    is_bwd = section_kind in {
        SectionKind.BACKWARD,
        SectionKind.FUSED_BACKWARD_UPDATE,
    }

    if logical_idx == 0 and not is_bwd:
        return True

    if not is_bwd:
        return (
            microbatch_idx,
            (SectionKind.FORWARD, logical_idx - 1),
        ) in done_tasks

    if (microbatch_idx, (SectionKind.FORWARD, logical_idx)) not in done_tasks:
        return False

    if logical_idx == num_logical_stages - 1:
        return True

    next_bwd_task = (microbatch_idx, (SectionKind.BACKWARD, logical_idx + 1))
    next_fused_task = (
        microbatch_idx,
        (SectionKind.FUSED_BACKWARD_UPDATE, logical_idx + 1),
    )
    return next_bwd_task in done_tasks or next_fused_task in done_tasks


def serialize_unaligned_jaxpp_schedule(
    schedule, num_logical_stages: int
) -> PipelineSchedule:
    """Given a partitioned schedule that is unaligned (physical stages might
    have schedules with different lengths because idle timesteps have been
    omitted), serialize it into a flattened schedule while respecting data
    dependencies between stage."""

    done_tasks: set[Task] = set()

    # physical_idx -> ready tasks queue
    ready_tasks: list[SimpleQueue] = [SimpleQueue() for _ in range(len(schedule))]

    # physical_idx -> timestep being inspected
    curr_timesteps: list[list[int]] = [0 for _ in range(len(schedule))]

    # Serialization loop.
    flat_schedule = []

    while True:
        # Update ready_tasks.
        for physical_idx, curr_t in enumerate(curr_timesteps):
            if curr_t >= len(schedule[physical_idx]):
                continue

            task = schedule[physical_idx][curr_t]
            assert task is not None, "Unexpected None task in unaligned schedule."

            if task_is_ready(task, done_tasks, num_logical_stages):
                ready_tasks[physical_idx].put(task)
                curr_timesteps[physical_idx] += 1

        # Break if no tasks are ready and all curr_timesteps are past the end
        # of their respective schedule.
        no_ready_tasks = all(stage_queue.qsize() == 0 for stage_queue in ready_tasks)
        all_timesteps_done = all(
            curr_t >= len(schedule[physical_idx])
            for physical_idx, curr_t in enumerate(curr_timesteps)
        )
        if no_ready_tasks and all_timesteps_done:
            break

        # If we are not done but there are no ready tasks, we have deadlocked.
        assert not no_ready_tasks, (
            "Deadlock detected inside serialize_aligned_jaxpp_schedule."
        )

        # Add tasks to flat_schedule.
        last_physical_stage_added = -1
        if len(flat_schedule) > 0:
            _, (_, logical_stage_idx) = flat_schedule[-1]
            last_physical_stage_added = logical_stage_idx % len(schedule)

        for tmp in range(len(schedule)):
            physical_idx = (last_physical_stage_added + 1 + tmp) % len(schedule)
            if ready_tasks[physical_idx].qsize() > 0:
                task = ready_tasks[physical_idx].get()
                flat_schedule.append(task)
                done_tasks.add(task)

    return flat_schedule


def _replace_last_stage0_backward_with_fused(compute_schedule: PipelineSchedule):
    for idx in range(len(compute_schedule) - 1, -1, -1):
        microbatch_idx, (section_kind, logical_stage_idx) = compute_schedule[idx]
        if section_kind == SectionKind.BACKWARD and logical_stage_idx == 0:
            compute_schedule[idx] = (
                microbatch_idx,
                (SectionKind.FUSED_BACKWARD_UPDATE, logical_stage_idx),
            )
            return compute_schedule
    return compute_schedule


def _add_1f1b_boundary_sections(
    compute_schedule: PipelineSchedule,
    num_logical_stages: int,
) -> PipelineSchedule:
    init_tasks = [
        (None, (SectionKind.STAGE_INIT, logical_stage_idx))
        for logical_stage_idx in range(num_logical_stages)
    ]
    fused_compute_schedule = _replace_last_stage0_backward_with_fused(compute_schedule)
    update_tasks = [
        (None, (SectionKind.UPDATE, logical_stage_idx))
        for logical_stage_idx in range(1, num_logical_stages)
    ]
    return init_tasks + fused_compute_schedule + update_tasks


def _make_jaxpp_non_interleaved_1F1B_fwd_bwd_schedule(num_stages, num_mubatches):
    steps = num_mubatches + num_stages - 1
    schedule = [[None] * (steps * 2) for _ in range(num_stages)]

    stage_mubatch = [[0, 0] for _ in range(num_stages)]

    # Warmup.
    for step in range(num_stages):
        for stage_id in range(num_stages):
            if step >= stage_id:
                mubatch_idx = stage_mubatch[stage_id][0]
                if mubatch_idx >= 0 and mubatch_idx < num_mubatches:
                    schedule[stage_id][step] = (
                        mubatch_idx,
                        (SectionKind.FORWARD, stage_id),
                    )
                    stage_mubatch[stage_id][0] += 1

    # Steady stage + cooldown.
    for step in range(num_stages, 2 * steps):
        relative_step = step - num_stages
        for stage_id in range(num_stages):
            inv_stage = num_stages - stage_id - 1
            if relative_step >= inv_stage:
                fwd_or_bwd = 1 - (relative_step + inv_stage) % 2  # 1 = bwd.
                mubatch_idx = stage_mubatch[stage_id][fwd_or_bwd]
                if mubatch_idx >= 0 and mubatch_idx < num_mubatches:
                    section_kind = (
                        SectionKind.BACKWARD if fwd_or_bwd else SectionKind.FORWARD
                    )
                    schedule[stage_id][step] = (mubatch_idx, (section_kind, stage_id))
                    stage_mubatch[stage_id][fwd_or_bwd] += 1

    return serialize_aligned_jaxpp_schedule(schedule)


def _make_jaxpp_interleaved_1F1B_fwd_bwd_schedule(
    num_logical_stages,
    num_physical_stages,
    num_mubatches,
):
    assert num_logical_stages % num_physical_stages == 0

    logical_per_physical = num_logical_stages // num_physical_stages

    number_of_rounds = max(1, num_mubatches // num_physical_stages)
    microbatches_per_round, _ = divmod(num_mubatches, number_of_rounds)
    if _ != 0:
        raise ValueError("n_microbatches must be divisible by mpmd_dim")

    def get_rank_warmup_ops(physical_idx):
        # Warms up operations for last stage.
        warmups_ops_last_stage = (logical_per_physical - 1) * microbatches_per_round
        # Increment warmup operations by 2 for each hop away from the last stage.
        multiply_factor = 2
        warmup_ops = warmups_ops_last_stage + multiply_factor * (
            (num_physical_stages - 1) - physical_idx
        )

        # We cannot have more warmup operations than there are number of
        # microbatches, so cap it there.
        return min(warmup_ops, num_mubatches * logical_per_physical)

    def forward_stage_index(physical_idx, step):
        # Get the local index from 0 to n_local_stages-1
        local_index = (step // microbatches_per_round) % logical_per_physical
        return (local_index * num_physical_stages) + physical_idx

    def backward_stage_index(step, warmup_ops, physical_idx):
        local_index = (
            logical_per_physical
            - 1
            - ((step - warmup_ops) // microbatches_per_round) % logical_per_physical
        )
        return (local_index * num_physical_stages) + physical_idx

    def _tasks_for_rank(physical_idx):
        microbatch_ops = logical_per_physical * num_mubatches
        warmup_ops = get_rank_warmup_ops(physical_idx)
        fwd_bwd_ops = microbatch_ops - warmup_ops
        cooldown_ops = microbatch_ops - fwd_bwd_ops

        # (logical_idx, is_bwd) -> microbatch.
        task_mubatch = defaultdict[tuple[int, bool], int](lambda: 0)

        def next_task(logical_idx, is_bwd):
            mubatch = task_mubatch[(logical_idx, is_bwd)]
            section_kind = SectionKind.BACKWARD if is_bwd else SectionKind.FORWARD
            res = (mubatch, (section_kind, logical_idx))
            task_mubatch[(logical_idx, is_bwd)] += 1
            return res

        tasks = []

        # Warmup.
        for step in range(warmup_ops):
            tasks.append(
                next_task(forward_stage_index(physical_idx, step), is_bwd=False)
            )

        # Steady state.
        for step in range(warmup_ops, warmup_ops + fwd_bwd_ops):
            fwd_idx = forward_stage_index(physical_idx, step)
            bwd_idx = backward_stage_index(step, warmup_ops, physical_idx)
            fwd = next_task(fwd_idx, is_bwd=False)
            bwd = next_task(bwd_idx, is_bwd=True)

            # if fwd_idx != num_logical_stages - 1:
            tasks.extend([fwd, bwd])

        # Cooldown.
        for step in range(
            warmup_ops + fwd_bwd_ops, warmup_ops + fwd_bwd_ops + cooldown_ops
        ):
            tasks.append(
                next_task(
                    backward_stage_index(step, warmup_ops, physical_idx),
                    is_bwd=True,
                )
            )
        return tasks

    schedule = [
        _tasks_for_rank(physical_idx) for physical_idx in range(num_physical_stages)
    ]
    return serialize_unaligned_jaxpp_schedule(schedule, num_logical_stages)


def make_1F1B_schedule(
    num_logical_stages,
    num_physical_stages,
    num_mubatches,
):
    if num_logical_stages == num_physical_stages:
        schedule = _make_jaxpp_non_interleaved_1F1B_fwd_bwd_schedule(
            num_logical_stages, num_mubatches
        )
    else:
        schedule = _make_jaxpp_interleaved_1F1B_fwd_bwd_schedule(
            num_logical_stages, num_physical_stages, num_mubatches
        )

    return _add_1f1b_boundary_sections(schedule, num_logical_stages)


# ---- Generic schedule generator ---- #


def make_pipeline_schedule(
    num_logical_stages,
    num_physical_stages,
    num_microbatches,
    schedule_name,
) -> PipelineSchedule:
    assert schedule_name in {
        "gpipe",
        "1F1B",
    }, "Only GPipe and 1F1B schedules are currently supported."

    if schedule_name == "1F1B":
        return make_1F1B_schedule(
            num_logical_stages,
            num_physical_stages,
            num_microbatches,
        )

    return make_gpipe_schedule(num_microbatches, num_logical_stages)
