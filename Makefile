# FinanceHarness — headless entrypoints. `make help` lists targets.
.DEFAULT_GOAL := help
.PHONY: help install run serve mcp mcp-http web web-install web-build app

# the service port (override e.g. `make serve PORT=9000`)
PORT ?= 8080
# the MCP-over-HTTP port (stdio needs no port)
MCP_PORT ?= 8765

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | sort

install:  ## install dependencies into the environment
	uv sync

run:  ## one-shot research: make run Q="your question"
	uv run python main.py -p "$(Q)"

serve:  ## run the HTTP+SSE service on $(PORT) (also serves web/dist if built)
	uv run fh serve --port $(PORT)

mcp:  ## serve the harness to MCP clients over stdio (Claude Desktop, IDEs)
	uv run fh mcp

mcp-http:  ## serve the harness over MCP streamable HTTP on $(MCP_PORT)
	uv run fh mcp --http --port $(MCP_PORT)

web-install:  ## install the web console's dependencies
	cd web && npm install

web:  ## run the web console's dev server (expects `make serve` alongside)
	cd web && npm run dev

web-build:  ## build the web console into web/dist (then `make serve` serves it)
	cd web && npm run build

app: web-build serve  ## build the console, then serve API + UI from one process
