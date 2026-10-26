# The controller under UAT — issue #293 increment 1.
#
# The published controller image, with THIS branch's Python AND THIS branch's
# dashboard compiled into it, because a console-error check that renders the
# released bundle cannot catch a regression in the bundle under review. That is
# the whole value of the job: `controller-imports` layers the Python only, and
# deliberately — /app/static/dashboard.js in the image is what the last release
# shipped, so a job that only copies *.py walks last month's dashboard and
# reports on it.
#
# Same overlay shape as controller-imports, plus the static sources the browser
# pass has to see and the esbuild run that compiles them. index.html is in the
# list because the walker enters at `/`, so the landing state machine is this
# branch's too; the released one would hide a regression there.
#
# The static files are named one by one rather than `COPY controller/static/`
# because /app/static also holds vendor/ — React, xterm and the fonts, fetched
# by the image build — and mounting or replacing the directory hides them. The
# dashboard then loads and every tab renders blank, which is a worse check than
# no check.
#
# PINNED BY DIGEST, and the same digest .github/workflows/ci.yml's
# controller-imports job pins, for the reason that job gives in full: a floating
# tag means this job's baseline can move without a PR, so a red check stops
# being evidence about the PR that is red. Currently 2.20.2 — bump BOTH pins
# together when a GA release changes dependencies; the symptom is this image
# failing to import something newly added to requirements.txt, which is a true
# signal about a stale pin. Get the digest with:
#
#   docker buildx imagetools inspect ghcr.io/wilbowes/echomuse-controller:latest
#
# esbuild is the version and the flags controller/Dockerfile uses, from the same
# pinned npm tarball, so the bundle compiled here is the bundle that ships.
FROM ghcr.io/wilbowes/echomuse-controller@sha256:3df932996b3c9ac6fc583eaf40aba904b9a86bf5e531169515ce49dd0c5216af
COPY controller/*.py /app/
COPY controller/static/dashboard.jsx controller/static/dashboard.html controller/static/index.html controller/static/favicon.svg /app/static/
RUN EARCH="$(dpkg --print-architecture)" && case "$EARCH" in amd64) EPKG=linux-x64 ;; arm64) EPKG=linux-arm64 ;; *) echo "unsupported arch: $EARCH" >&2; exit 1 ;; esac \
 && curl -sfL -o /tmp/esbuild.tar.gz "https://registry.npmjs.org/@esbuild/${EPKG}/-/${EPKG}-0.20.2.tgz" \
 && tar -xz -C /tmp -f /tmp/esbuild.tar.gz \
 && install -m755 /tmp/package/bin/esbuild /usr/local/bin/esbuild \
 && esbuild /app/static/dashboard.jsx --bundle=false --jsx=transform --jsx-factory=React.createElement --jsx-fragment=React.Fragment --outfile=/app/static/dashboard.js \
 && rm -rf /tmp/esbuild.tar.gz /tmp/package /usr/local/bin/esbuild
