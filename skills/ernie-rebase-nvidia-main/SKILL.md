---
name: ernie-rebase-nvidia-main
description: Rebase the ERNIE fork's downstream commits onto a freshly-synced upstream NVIDIA main, resolving conflicts, keeping the Megatron-LM submodule in lockstep with main, tagging the previous stable point, and backing up the branch being replaced.
when_to_use: Periodically re-baselining the ERNIE fork on newer upstream NVIDIA/Megatron-Bridge main; 'rebase on nvidia main', 'cut a new rebase-nvidia-main-YYYYMMDD branch', 'sync fork and replay ERNIE commits', 'bump to latest upstream'.
---

# ERNIE Rebase onto NVIDIA main

This repo is a downstream fork (`origin` = a GitHub fork of NVIDIA/Megatron-Bridge).
Periodically the small set of ERNIE commits is replayed on top of a newer upstream
`main`. The result is a new `rebase-nvidia-main-<YYYYMMDD>` branch, the previous
stable point is tagged, and the branch being overwritten is backed up first.

Naming convention: `rebase-nvidia-main-<YYYYMMDD>` where the date is the day the
new upstream `main` was synced (matches the fork sync date, not necessarily today).

## Environment

- **No direct internet.** Any remote git op (fetch/push/submodule clone) needs the proxy:
  ```bash
  export http_proxy="http://cmcproxy:...@10.251.112.50:8128"
  export https_proxy="http://cmcproxy:...@10.251.112.50:8128"
  ```
  The user supplies the proxy credential when they want you to reach the network. Do
  not hardcode it into files.
- **Megatron-LM submodule normally follows `main`.** The ERNIE commits usually don't
  touch `3rdparty/Megatron-LM`, so on rebase its gitlink just advances to whatever the
  new `main` points at (verify with `git log -p "$BASE".."$STABLE" -- 3rdparty/Megatron-LM`).
  The fork *may* patch the submodule when it genuinely needs to — that's allowed — but
  it goes through the normal flow (commit inside the submodule → bump the gitlink here),
  and on rebase such an intentional change must be **preserved**, not overwritten by
  main. Don't blindly follow main without checking whether ERNIE changed it.
- **Token hygiene:** if the user pastes a GitHub token to enable a push, use it for
  that push only and remind them to rotate it afterward. Never write it to a file.

## Preconditions

1. The current stable branch is verified-good (the user says "this branch is stable").
2. The user has already **synced their fork's `main`** on GitHub. Confirm this — the
   whole point is to rebase onto the *new* upstream state, so `origin/main` must be
   fresh before you fetch.

## Procedure

### 1. Fetch the freshly-synced main

```bash
git fetch origin main            # needs proxy
git log --oneline -5 origin/main # confirm it advanced to the expected upstream state
```

### 2. Identify the ERNIE commit set to replay

The rebase base is the merge-base of the current stable branch and the *new* main —
this equals the upstream commit the previous rebase was built on:

```bash
STABLE=<current-stable-commit>          # e.g. the tip you were told is stable
BASE=$(git merge-base "$STABLE" origin/main)
git log --oneline "$BASE".."$STABLE"    # the ERNIE commits — expect a small number
git log --format="%an" "$BASE".."$STABLE" | sort | uniq -c   # sanity: all ERNIE authors
```

If this shows more commits than expected, or non-ERNIE authors, STOP and confirm with
the user before rebasing — the base may be wrong.

### 3. Create the new branch and rebase

```bash
git branch -f rebase-nvidia-main-<YYYYMMDD> "$STABLE"
git checkout rebase-nvidia-main-<YYYYMMDD>
git rebase --onto origin/main "$BASE" rebase-nvidia-main-<YYYYMMDD>
```

### 4. Resolve conflicts

Conflicts are typically in the training loop / checkpointing files the ERNIE commits
touch (`train.py`, `checkpointing.py`, `train_utils.py`, `common_utils.py`, ...).

Resolution principle: **keep upstream's new structure, re-apply the ERNIE intent on
top of it.** Concretely:
- When upstream refactored a block (e.g. added MSC/`MultiStorageClientFeature`
  branches, renamed helpers), take the upstream version and re-inject the ERNIE
  additions (local-rank cleanup guards, `_CHECKPOINT_CLEANUP_LOCK`,
  `num_total_tokens_in_batch`, extra `training_log` kwargs, etc.) into it.
- Before wiring an ERNIE kwarg into a call, verify the callee still accepts it
  (`grep -n "def <fn>"` and check the signature). Signatures drift upstream.
- Prefer keeping *both* an upstream param and an ERNIE param when both are still valid
  and in scope — don't drop one to silence the conflict.

After each file:
```bash
grep -rn '^<<<<<<<\|^>>>>>>>' <file>          # no markers left
python -m py_compile <file>                    # syntax check
git add <file>                                 # NEVER `git add 3rdparty/Megatron-LM` here
GIT_EDITOR=true git rebase --continue
```

Do **not** `git add` the `3rdparty/Megatron-LM` gitlink during conflict resolution —
handle the submodule deliberately in step 5.

### 5. Move the submodule gitlink to follow main

The rebased branch's index should already record main's Megatron-LM gitlink (the
ERNIE commits normally don't touch the submodule — verify with
`git log -p "$BASE".."$STABLE" -- 3rdparty/Megatron-LM`). The **submodule follows
main.** Align the working tree to the index:

```bash
git rev-parse HEAD:3rdparty/Megatron-LM                       # target gitlink (from main)
(cd 3rdparty/Megatron-LM && git rev-parse HEAD)               # current checkout
git submodule update --init 3rdparty/Megatron-LM             # may need proxy if commit not local
```

Confirm both hashes match and `git status` no longer shows the submodule as modified.
If the ERNIE commits *did* intentionally change the submodule, surface that to the
user instead of blindly following main.

### 6. Verify

```bash
git log --oneline -6                     # new main tip + replayed ERNIE commits on top
grep -rn '^<<<<<<<\|^>>>>>>>' src/        # no residual markers
```

Run any lint/tests the change warrants per the repo's testing skill.

## Tag the previous stable point + back up the overwritten branch

When promoting the new branch and retiring the old rolling branch (e.g. `wip`):

1. **Tag** the stable commit so it's recoverable by name:
   ```bash
   git tag -a <stable-tag-name> <stable-commit> -m "稳定版本: <name> (<short>)"
   ```
   Double-check the date/name with the user — it's easy to typo the year.
2. **Back up** any branch you're about to force-overwrite, with BOTH a branch and a
   tag (double insurance), dated to when that branch was current:
   ```bash
   git branch  wip-backup-<YYYYMMDD>     origin/wip
   git tag -a  wip-backup-tag-<YYYYMMDD> origin/wip -m "备份: force-push 前的旧 origin/wip"
   ```
   Use distinct branch vs tag names (`...-<date>` vs `...-tag-<date>`) to avoid git's
   "refname is ambiguous" warning.
3. **Force-push safely** with an explicit expected old value:
   ```bash
   git push origin wip:wip --force-with-lease=wip:<old-sha>
   ```

Always push the backups *before* the force-push, and verify remote with
`git ls-remote origin <refs...>` afterward.

## Pushing (proxy + token)

Fetch/ls-remote work with just the proxy. Pushes may fail with
`could not read Username for 'https://github.com'` — the fork uses HTTPS with no
stored credential. Push via a one-off tokenized URL (token from the user, used inline,
never written to disk):

```bash
git push "https://<TOKEN>@github.com/<owner>/Megatron-Bridge.git" <ref>
```

## Guardrails

- Confirm merge direction and the exact ERNIE commit set with the user before rebasing
  when anything looks off (unexpected commit count, wrong base, submodule changes).
- Force-pushing a rolling branch discards its unique commits — back up first, and
  state plainly which commits are being dropped.
- The submodule normally follows main, but if the ERNIE commits intentionally patched
  `3rdparty/Megatron-LM`, preserve that change rather than resetting to main's gitlink.
