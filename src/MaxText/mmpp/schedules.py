from queue import SimpleQueue
from functools import partial
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# (mubatch_idx, logical_stage_idx, is_bwd)
Task = tuple[int, int, bool]

# physical_idx -> list[Task]
Schedule = list[list[Task | None]]

# Topologically sorted Schedule, omitting idle "None" tasks.
FlattenedSchedule = list[Task]


# ==== Plotting utility to check schedules ==== #


def plot_schedule(
    schedule: Schedule,
    filename: str,
    num_logical_stages=None,
):
    """Plot and save a pipeline schedule."""
    num_stages = len(schedule)
    if num_stages == 0:
        print("Empty schedule")
        return

    num_steps = len(schedule[0])

    # Create figure and axes.
    # Adjust width based on number of steps to prevent squashing
    fig, ax = plt.subplots(figsize=(max(10, num_steps * 0.6), max(5, num_stages * 1.0)))

    # Set limits and invert y axis so stage 0 is top.
    ax.set_xlim(0, num_steps)
    ax.set_ylim(0, num_stages)
    ax.invert_yaxis()

    # --- COLOR PALETTE ---
    # List of (Forward Color, Backward Color) tuples for different "chunks" (repeats).
    # Chunk 0: Blue / Green
    # Chunk 1: Orange / Red
    # Chunk 2: Purple / Pink
    # Chunk 3: Cyan / Yellow
    chunk_colors = [
        ("#ADD8E6", "#90EE90"),  # Light Blue, Light Green
        ("#FFDAB9", "#FA8072"),  # Peach Puff, Salmon
        ("#D8BFD8", "#FFB6C1"),  # Thistle, Light Pink
        ("#E0FFFF", "#F0E68C"),  # Light Cyan, Khaki
    ]

    for stage_id, stage_timeline in enumerate(schedule):
        for step, task in enumerate(stage_timeline):
            if task is None:
                continue

            mubatch_idx, logical_stage_idx, is_bwd = task

            # Calculate which "repeat" or "chunk" this is.
            # In interleaved, physical stage 0 might handle logical stages 0, 4, 8...
            chunk_idx = logical_stage_idx // num_stages

            # Select color pair based on chunk index (cycling if we exceed palette size).
            fwd_c, bwd_c = chunk_colors[chunk_idx % len(chunk_colors)]
            color = bwd_c if is_bwd else fwd_c

            # Construct Label.
            # F/B{microbatch} (c{chunk})
            type_char = "B" if is_bwd else "F"
            label = f"{type_char}{mubatch_idx}\n(c{chunk_idx})"

            # Rectangle(xy, width, height).
            rect = patches.Rectangle(
                (step, stage_id), 1, 1, linewidth=1, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)

            # Add text.
            ax.text(
                step + 0.5,
                stage_id + 0.5,
                label,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,  # Slightly smaller font to fit the extra text
                color="black",
            )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Physical Stage")

    # Set ticks.
    ax.set_yticks([i + 0.5 for i in range(num_stages)])
    ax.set_yticklabels([f"Phy {i}" for i in range(num_stages)])

    ax.set_xticks([i + 0.5 for i in range(num_steps)])
    ax.set_xticklabels(range(num_steps), rotation=90, fontsize=8)

    # Grid lines.
    ax.grid(True, which="both", color="gray", linestyle="--", linewidth=0.5)

    plt.title(filename.replace(".png", ""))
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved schedule plot to {filename}")


# ==== JaxPP Implementations ==== #

# The implementations below are lightly modified versions of JaxPP schedules,
# and are taken from here:
# https://github.com/NVIDIA/jaxpp/blob/5fde1d2bc2e4ecaeea05dce08b931c5a790b9619/src/jaxpp/schedules.py


# ---- Schedule serialization methods ---- #

# mmpp expects "flattened" schedules that look like list[Task] instead of the
# list[list[Task]] returned by JaxPP (one schedule per physical stage). These
# functions flatten schedules from the latter to the former.


def serialize_aligned_jaxpp_schedule(aligned_schedule: Schedule) -> FlattenedSchedule:
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
    mubatch_idx, logical_idx, is_bwd = task

    if logical_idx == 0 and not is_bwd:
        return True

    if not is_bwd:
        return (mubatch_idx, logical_idx - 1, False) in done_tasks

    if (mubatch_idx, logical_idx, False) not in done_tasks:
        return False

    if logical_idx == num_logical_stages - 1:
        return True

    return (mubatch_idx, logical_idx + 1, True) in done_tasks


def serialize_unaligned_jaxpp_schedule(
    schedule: Schedule, num_logical_stages: int
) -> FlattenedSchedule:
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
            last_physical_stage_added = flat_schedule[-1][1] % len(schedule)

        for tmp in range(len(schedule)):
            physical_idx = (last_physical_stage_added + 1 + tmp) % len(schedule)
            if ready_tasks[physical_idx].qsize() > 0:
                task = ready_tasks[physical_idx].get()
                flat_schedule.append(task)
                done_tasks.add(task)

    return flat_schedule


def align_jaxpp_schedule(unaligned_schedule, num_logical_stages):
    """Given a partitioned schedule that is unaligned (physical stages might
    have schedules with different lengths because idle timesteps have been
    omitted), align it by greedily executing task as soon as they are ready."""

    num_physical_stages = len(unaligned_schedule)

    done_tasks: set[Task] = set()
    aligned_schedule = [[] for _ in range(num_physical_stages)]
    curr_timesteps = [0 for _ in range(num_physical_stages)]

    while True:
        ready_tasks = [None for _ in range(num_physical_stages)]
        for physical_idx in range(num_physical_stages):
            curr_t = curr_timesteps[physical_idx]
            if curr_t >= len(unaligned_schedule[physical_idx]):
                continue
            task = unaligned_schedule[physical_idx][curr_t]
            assert task is not None, "Unexpected idle task in input schedule."
            if task_is_ready(task, done_tasks, num_logical_stages):
                ready_tasks[physical_idx] = task
                curr_timesteps[physical_idx] += 1

        # print(f"READY TASKS AT t={len(aligned_schedule[0])}: {ready_tasks}")

        forward_progress = not all(map(lambda task: task is None, ready_tasks))

        if not forward_progress and all(
            curr_t >= len(unaligned_schedule[physical_idx])
            for physical_idx, curr_t in enumerate(curr_timesteps)
        ):
            break

        if not forward_progress:
            print("Aligned schedule so far:")
            print(aligned_schedule)
            print()

            print("curr_timesteps:")
            print(curr_timesteps)
            print()

            print("Next tasks:")
            print(
                tuple(
                    f"physical_idx={physical_idx}, curr_t={curr_t}, task={unaligned_schedule[physical_idx][curr_t]}"
                    for physical_idx, curr_t in enumerate(curr_timesteps)
                )
            )

            raise ValueError("Deadlock detected in input schedule.")

        for physical_idx, task in enumerate(ready_tasks):
            aligned_schedule[physical_idx].append(task)
            if task is not None:
                done_tasks.add(task)

    return aligned_schedule


# ---- JaxPP GPipe ---- #


def _make_jaxpp_interleaved_gpipe_schedule(
    num_logical_stages,
    num_physical_stages,
    num_mubatches,
    flatten=True,
):
    # This is hardcoded for 2 logical / physical stage.
    assert num_logical_stages // num_physical_stages == 2, (
        "_make_jaxpp_interleaved_gpipe_schedule only works for 2 logical / physical stage."
    )

    FWD, BWD = 0, 1
    half_steps = num_mubatches * 2 + num_physical_stages - 1
    n_steps = half_steps * 2
    schedule = [([None] * n_steps) for _ in range(num_physical_stages)]

    # stage_mubatch[physical_idx][0] =
    #   (next_fwd_logical_idx, next_mubatch, counter).
    # stage_mubatch[physical_idx][1] =
    #   (next_bwd_logical_idx, next_mubatch, counter).
    stage_mubatch = list[list[tuple[int, int, int]]](
        [(0, dim_id, 0), (0, dim_id + num_physical_stages, 0)]
        for dim_id in range(num_physical_stages)
    )

    # If physical_idx just finished a task, what is the next task it should do?
    # fwd_or_bwd: If fwd_or_bwd == FWD, then we are requesting the next fwd
    #   stage, and updating the fwd entry in stage_mubatch.
    # count: The number of times physical_idx has performed fwd_or_bwd so far.
    def get_next(physical_idx, fwd_or_bwd, value, count):
        assert value == 1, f"Expect an update by increasing 1, but `{value}` found."
        count += value

        # If rem_stages < num_physical_stages, we are now on the first chunk. Else,
        # we are now on the second chunk.
        rem_stages = count % num_logical_stages

        is_fwd = fwd_or_bwd == 0

        # Next logical stage we have to execute.
        stage_id = (
            (physical_idx if is_fwd else physical_idx + num_physical_stages)
            if rem_stages < num_physical_stages
            else (physical_idx + num_physical_stages if is_fwd else physical_idx)
        )

        # Next mubatch we have to execute.
        # count % num_physical_stages is the offset into the 'wave' of
        # microbatches being processed for the current chunk. One chunk has
        # size num_physical. We do num_chunks of these waves before moving to
        # the next wave of microbatches. This gives
        # (count // (num_chunks * num_physical)) * num_physical as the 'start'
        # of the wave.
        mubatch_idx = (count // num_logical_stages) * num_physical_stages + (
            count % num_physical_stages
        )
        return (mubatch_idx, stage_id, count)

    # fwd: the first half.
    for step in range(half_steps):
        for physical_idx in range(num_physical_stages):
            if step >= physical_idx:
                # FIX: Unpack as (Stage, Mubatch, Count)
                mubatch_idx, stage_id, count = stage_mubatch[physical_idx][0]

                if mubatch_idx >= 0 and mubatch_idx < num_mubatches:
                    schedule[physical_idx][step] = (mubatch_idx, stage_id, False)
                    stage_mubatch[physical_idx][FWD] = get_next(
                        physical_idx, FWD, 1, count
                    )

    # bwd: the second half.
    for step in range(half_steps, n_steps):
        relative_step = step - half_steps
        for physical_idx in range(num_physical_stages):
            inv_step = num_physical_stages - physical_idx - 1
            if relative_step >= inv_step:
                # FIX: Unpack as (Stage, Mubatch, Count)
                mubatch_idx, stage_id, count = stage_mubatch[physical_idx][BWD]

                if mubatch_idx >= 0 and mubatch_idx < num_mubatches:
                    schedule[physical_idx][step] = (mubatch_idx, stage_id, True)
                    stage_mubatch[physical_idx][BWD] = get_next(
                        physical_idx, BWD, 1, count
                    )

    if flatten:
        schedule = serialize_unaligned_jaxpp_schedule(schedule, num_logical_stages)

    return schedule


def _make_jaxpp_non_interleaved_gpipe_schedule(
    num_stages,
    num_mubatches,
    flatten=True,
):
    steps = num_mubatches + num_stages - 1
    schedule = [[None] * (steps * 2) for _ in range(num_stages)]
    for step in range(steps):
        for stage_id in range(num_stages):
            mubatch_idx = step - stage_id
            if mubatch_idx >= 0 and mubatch_idx < num_mubatches:
                schedule[stage_id][step] = (mubatch_idx, stage_id, False)

    for step in range(steps, steps * 2):
        for stage_id in reversed(range(num_stages)):
            mubatch_idx = (step - steps) - (num_stages - stage_id - 1)
            if mubatch_idx >= 0 and mubatch_idx < num_mubatches:
                schedule[stage_id][step] = (mubatch_idx, stage_id, True)

    if flatten:
        schedule = serialize_aligned_jaxpp_schedule(schedule)

    return schedule


def make_jaxpp_gpipe_schedule(
    num_logical_stages,
    num_physical_stages,
    num_mubatches,
    flatten=True,
):
    assert num_logical_stages % num_physical_stages == 0

    if num_logical_stages == num_physical_stages:
        return _make_jaxpp_non_interleaved_gpipe_schedule(
            num_logical_stages, num_mubatches, flatten
        )

    # TODO: Verify this is actually the case?
    assert num_logical_stages // num_physical_stages == 2, (
        "JaxPP Interleaved Gpipe only supports 2 repeats."
    )

    return _make_jaxpp_interleaved_gpipe_schedule(
        num_logical_stages, num_physical_stages, num_mubatches, flatten
    )


# ---- JaxPP 1F1B ---- #


def _make_jaxpp_non_interleaved_1F1B_schedule(num_stages, num_mubatches, flatten=True):
    steps = num_mubatches + num_stages - 1
    schedule = [[None] * (steps * 2) for _ in range(num_stages)]

    stage_mubatch = [[0, 0] for _ in range(num_stages)]

    # Warmup.
    for step in range(num_stages):
        for stage_id in range(num_stages):
            if step >= stage_id:
                mubatch_idx = stage_mubatch[stage_id][0]
                if mubatch_idx >= 0 and mubatch_idx < num_mubatches:
                    schedule[stage_id][step] = (mubatch_idx, stage_id, False)
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
                    schedule[stage_id][step] = (mubatch_idx, stage_id, bool(fwd_or_bwd))
                    stage_mubatch[stage_id][fwd_or_bwd] += 1

    if flatten:
        schedule = serialize_aligned_jaxpp_schedule(schedule)
    return schedule


def _make_jaxpp_interleaved_1F1B_schedule(
    num_logical_stages,
    num_physical_stages,
    num_mubatches,
    flatten=True,
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

        # (logical_idx, is_bwd) -> mubatch.
        task_mubatch = defaultdict[tuple[int, bool], int](lambda: 0)

        def next_task(logical_idx, is_bwd):
            mubatch = task_mubatch[(logical_idx, is_bwd)]
            res = (mubatch, logical_idx, is_bwd)
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
    if flatten:
        schedule = serialize_unaligned_jaxpp_schedule(schedule, num_logical_stages)

    return schedule


def make_jaxpp_1F1B_schedule(
    num_logical_stages, num_physical_stages, num_mubatches, flatten=True
):
    if num_logical_stages == num_physical_stages:
        return _make_jaxpp_non_interleaved_1F1B_schedule(
            num_logical_stages, num_mubatches, flatten
        )

    return _make_jaxpp_interleaved_1F1B_schedule(
        num_logical_stages, num_physical_stages, num_mubatches, flatten
    )


# ==== CUSTOM IMPLEMENTATIONS ==== #


def make_gpipe_schedule(
    num_logical_stages, num_physical_stages, num_mubatches, flatten=True
):
    assert num_logical_stages % num_physical_stages == 0, (
        "num_physical_stages does not divide num_logical_stages."
    )
    num_logical_per_physical = num_logical_stages // num_physical_stages
    num_rounds = ((num_mubatches - 1) // num_physical_stages) + 1

    num_steps = 2 * (num_mubatches * num_logical_per_physical + num_physical_stages - 1)
    schedule = [[None for _ in range(num_steps)] for _ in range(num_physical_stages)]

    for physical_idx in range(num_physical_stages):
        curr_timestep = 0

        # Wait for burn-in.
        curr_timestep += physical_idx

        # Forwards.
        for round_idx in range(num_rounds):
            mubatch_start_idx = round_idx * num_physical_stages
            mubatch_end_idx = min(
                mubatch_start_idx + num_physical_stages, num_mubatches
            )

            for my_logical_idx in range(num_logical_per_physical):
                logical_idx = (num_physical_stages * my_logical_idx) + physical_idx

                for mubatch_idx in range(mubatch_start_idx, mubatch_end_idx):
                    schedule[physical_idx][curr_timestep] = (
                        mubatch_idx,
                        logical_idx,
                        False,
                    )
                    curr_timestep += 1

        fwd_end_timestep = curr_timestep

        # Wait for bubble.
        my_bubble_time = 2 * (num_physical_stages - physical_idx - 1)
        curr_timestep += my_bubble_time

        # Backwards.
        for fwd_timestep_idx in reversed(range(physical_idx, fwd_end_timestep)):
            mubatch_idx, logical_idx, is_bwd = schedule[physical_idx][fwd_timestep_idx]
            assert not is_bwd
            schedule[physical_idx][curr_timestep] = (
                mubatch_idx,
                logical_idx,
                True,
            )
            curr_timestep += 1

    if flatten:
        schedule = serialize_aligned_jaxpp_schedule(schedule)

    return schedule


if __name__ == "__main__":
    # # JaxPP GPipe, 1 logical / physical stage.
    # plot_schedule(
    #     make_jaxpp_gpipe_schedule(
    #         num_logical_stages=4, num_physical_stages=4, num_mubatches=8, flatten=False
    #     ),
    #     "schedule-plots/jaxpp_gpipe-nu=8-nc=1.png",
    # )

    # # JaxPP GPipe, 2 logical / physical stage.
    # plot_schedule(
    #     make_jaxpp_gpipe_schedule(
    #         num_logical_stages=8, num_physical_stages=4, num_mubatches=8, flatten=False
    #     ),
    #     "schedule-plots/jaxpp_gpipe-nu=8-nc=2.png",
    # )

    # # JaxPP GPipe, 2 logical / physical stage.
    # plot_schedule(
    #     make_jaxpp_gpipe_schedule(
    #         num_logical_stages=8, num_physical_stages=4, num_mubatches=16, flatten=False
    #     ),
    #     "schedule-plots/jaxpp_gpipe-nu=16-nc=2.png",
    # )

    # JaxPP 1F1B, 1 logical / physical stage.
    plot_schedule(
        make_jaxpp_1F1B_schedule(
            num_logical_stages=4, num_physical_stages=4, num_mubatches=8, flatten=False
        ),
        "schedule-plots/jaxpp_1F1B-nu=8-nc=1.png",
        num_logical_stages=4,
    )

    # JaxPP 1F1B, 2 logical / physical stage.
    plot_schedule(
        align_jaxpp_schedule(
            make_jaxpp_1F1B_schedule(
                num_logical_stages=8,
                num_physical_stages=4,
                num_mubatches=8,
                flatten=False,
            ),
            num_logical_stages=8,
        ),
        "schedule-plots/jaxpp_1F1B-nu=8-nc=2.png",
        num_logical_stages=8,
    )

    # JaxPP 1F1B, 2 logical / physical stage.
    plot_schedule(
        align_jaxpp_schedule(
            make_jaxpp_1F1B_schedule(
                num_logical_stages=8,
                num_physical_stages=4,
                num_mubatches=16,
                flatten=False,
            ),
            num_logical_stages=8,
        ),
        "schedule-plots/jaxpp_1F1B-nu=16-nc=2.png",
        num_logical_stages=8,
    )

    # JaxPP 1F1B, 4 logical / physical stage.
    plot_schedule(
        align_jaxpp_schedule(
            make_jaxpp_1F1B_schedule(
                num_logical_stages=16,
                num_physical_stages=4,
                num_mubatches=16,
                flatten=False,
            ),
            num_logical_stages=16,
        ),
        "schedule-plots/jaxpp_1F1B-nu=16-nc=4.png",
        num_logical_stages=16,
    )
