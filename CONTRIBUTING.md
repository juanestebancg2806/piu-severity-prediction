# Contributing

All team members are collaborators of this repository. Work directly on it (clone, branch, pull request). Do not fork it.

## First-time setup

1. Accept the collaborator invitation sent by GitHub (check your email).
2. Clone the repository and install the environment:

   ```bash
   git clone https://github.com/juanestebancg2806/piu-severity-prediction.git
   cd piu-severity-prediction
   uv sync
   ```

3. Download the data into `data/raw/` as described in [data/README.md](data/README.md). The data is never committed.

## Daily workflow

1. Update your local `main`:

   ```bash
   git checkout main
   git pull
   ```

2. Create a branch for your task (short, descriptive name):

   ```bash
   git checkout -b feature/eda-target
   ```

3. Work and save your changes in small commits:

   ```bash
   git status                 # review what changed
   git add notebooks/01_eda.ipynb
   git commit -m "Add target distribution analysis"
   ```

4. Push your branch:

   ```bash
   git push -u origin feature/eda-target
   ```

5. Open a pull request on GitHub from your branch to `main` and ask a teammate to review it.
6. After it is merged, go back to step 1 before starting a new task.

## Rules

- Never commit files from `data/`. Run `git status` before every commit to check.
- Clear row-level outputs (`head()`, `sample()`, etc.) from notebooks before committing.
- Do not edit the same notebook as another teammate at the same time; coordinate first.
- If you get a merge conflict and are not sure how to solve it, stop and ask the team instead of forcing a push.
- Full project rules (privacy, leakage prevention, architecture, code style) are in [AGENTS.md](AGENTS.md).
