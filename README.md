# mla

Published docs: https://brokkoli71.github.io/mla/

## Build docs locally

1. Sync dependencies (includes docs group by default):
   `uv sync`
2. Build Sphinx HTML:
   `uv run sphinx-build -b html docs/source docs/build/html`
3. Open the result:
   `docs/build/html/index.html`
