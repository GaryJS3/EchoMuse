# Automated dashboard UAT

The machine-checked half of #293. The release rig in `tools/uat/` needs a
rooted Echo and a person; this needs neither, and runs on every PR against the
image built from the branch under review.

| | |
|---|---|
| `console_check.mjs` | every tab in headless Chromium, failing on a console error, an uncaught exception or a failed same-origin request (increment 1) |
| `Dockerfile` | the controller that check walks: the published image with this branch's Python **and** this branch's compiled dashboard layered over it |

`.github/workflows/ci.yml`'s `dashboard-console-errors` job boots the image,
reads the first-run bootstrap token out of the controller log and runs the
check inside the digest-pinned Playwright container. Screenshot baselines
(increment 2), axe (increment 3) and functional control coverage (increment 4)
are not here yet.

Run it by hand against a controller you already have up: the invocation and
its environment variables are in `console_check.mjs`'s header.
