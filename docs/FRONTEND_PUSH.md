# Push the frontend changes to Front-end-Update

Checked repository state: `origin` is
`https://github.com/Bensonwu929/RSMeM-Studio.git`. The local branch is `master`;
remote `master` and `Front-end-Update` both pointed to
`960a1d8ca15952964d20c16f66a516b321328c8b` when checked. These changes have not been
committed or pushed by the assistant.

Run the following from Terminal. The branch switch keeps the current frontend
edits, then the commit records them on `Front-end-Update`.

```bash
cd "/Users/kbaijin/Desktop/Agent Studio ModelScope/RSMeM-Studio"
git fetch origin
git switch -c Front-end-Update --track origin/Front-end-Update
git add app.py studio_ui.py studio_i18n.py assets/ docs/UI_REDESIGN.md docs/FRONTEND_PUSH.md scripts/build_studio_previews.py tests/test_studio_ui.py tests/test_studio_language.py
git diff --cached --stat
git commit -m "Make remote-sensing analysis easier to understand and use" -m "Put queries first, explain the workflow, preserve imagery proportions, and offer bilingual interface copy." -m "Scope-risk: narrow" -m "Tested: Python regressions and local browser interaction checks"
git push -u origin Front-end-Update
```

If the local branch has already been created, replace the `git switch -c ...`
line with `git switch Front-end-Update`. The remote branch can be reviewed at
https://github.com/Bensonwu929/RSMeM-Studio/tree/Front-end-Update after pushing.

The push publishes commits to that branch. Updating a separately deployed
ModelScope Studio still depends on that deployment's repository/branch settings.
