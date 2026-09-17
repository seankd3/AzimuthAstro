# Working in AzimuthAstro (for the agent)

The point of every rule here is fewer tokens per night. Measured on the first two nights: polling
cost ~60 calls, looking at images ~20 reads at 1,500 tokens each, and ad-hoc Python probes ~15 runs.

## Run a night
- `azastro process <work> <name> <cr3_folder>` is the whole night. Then ONE `azastro wait <work>`
  per ~9 minutes at most; its exit-3 line carries an ETA, so sleep to the ETA instead of polling.
- Every stage writes a `.done` marker and runs its gate; `azastro run --from S` redoes S onward,
  `--stills` re-renders only the stills after a colour or tone change, `--videos` only the videos.
- Windows multiprocessing in a detached run can lose a worker respawn late in a stage (`DuplicateHandle:
  Access is denied`, seen in register and three times in stack, with nothing else running). Pool stages
  must finish dropped work in-process (stack.py does); do not blame a concurrent job.
- `azastro stop <work>` kills the chain by its own pid tree. Never taskkill by a command-line regex
  (a stage name once matched `--to deliver` and killed the wrong chain).

## Judge a night without opening images
- `azastro probe <work>` first: colour of sky, stars and trails in camera space and as rendered,
  per-channel noise, star coverage, frames per pixel, clouds, fit residuals, tone, and the sky
  colour of every delivered still and video. Neutral is R/G 1.00 B/G 1.00; magenta is both above 1.
- `azastro report <work>` is the gates plus one contact sheet (mask overlay, treeline, composite).
  Read that one image, not the previews. Open a full still only to judge aesthetics.
- Every failure found by eye so far became a gate (gates.py). When you find one by eye, add its gate
  in the same commit.

## Change code
- Edit with the Edit tool. Multi-file patches: write the patch script with the Write tool, run it once;
  never inline Python in a heredoc with quotes in it (it breaks on the shell wrapper).
- `python smoke.py` (~2 min) before every commit; it runs hot..tone on a synthetic night with a known
  pole and passes gates. Commit identity is set locally; `git commit -m` is enough.
- One idea per file; a stage is a script that reads ASTRO_WORK. Add a stage by adding it to STAGES,
  the command map, and (if it can fail silently) gates.py.

## Data
- The engine directory `D:\AstroWork\engine` is a junction to this repo. Work dirs live in
  `D:\AstroWork\<night>`; the work drive must write fast (a night writes ~90 GB).
- Nothing under a work dir is ever committed; `docs/*.jpg` are the only images in the repo.
