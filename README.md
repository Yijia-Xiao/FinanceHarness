<p align="center">
  <img src="assets/FinanceHarness.svg" alt="FinanceHarness" width="60%">
</p>

<div align="center">
  <a href="https://www.readme-i18n.com/Yijia-Xiao/FinanceHarness?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/Yijia-Xiao/FinanceHarness?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/Yijia-Xiao/FinanceHarness?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/Yijia-Xiao/FinanceHarness?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/Yijia-Xiao/FinanceHarness?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/Yijia-Xiao/FinanceHarness?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/Yijia-Xiao/FinanceHarness?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/Yijia-Xiao/FinanceHarness?lang=zh">中文</a>
</div>

---

# FinanceHarness: Autonomous Financial Deep Research Framework

<p align="center">
  <a href="https://arxiv.org/pdf/2607.27853">
    <img src="assets/schema.svg" alt="The FinanceHarness stack: orchestration, capability, tools, runtime and model serving" width="100%">
  </a>
</p>

## Overview

**FinanceHarness is an autonomous financial deep research framework.** Ask a
question; FinanceHarness plans, gathers evidence, runs the analysis, and writes a
cited report.

- **Research and compute.** Web search and reading, company and market data, and
  valuation and risk computed from the gathered data, not guessed.
- **Reference-chaining.** One tool's output feeds the next by reference
  (`prev:<call_id>.<path>`), so a price series becomes a correlation with no
  retyping.
- **Progressive disclosure.** The core loop stays in the prompt; everything else
  waits in a catalog and loads on request, so breadth costs nothing until used.
- **Extensible without code.** A `SKILL.md` file composes existing tools into a
  reusable workflow. Drop one in and it is discovered.
- **Grounded.** Every figure traces back to the tool or source it came from.

## Get Started

### Prerequisites

FinanceHarness needs [`uv`](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install

```bash
git clone https://github.com/Yijia-Xiao/FinanceHarness.git
cd FinanceHarness
uv tool install .  # installs the `financeharness` command (alias `fh`)
```

From a source checkout, `uv sync` sets up the environment and
`uv run python main.py ...` (or `uv run fh ...`) runs the same CLI without
installing.

### Configure a backbone

Backbones are named by the provider or runtime that serves them. The default is
`gemini` (a cloud backbone that requires only an API key). Configure whichever
suits the machine you are on:

| Backbone | Serves | Infra | Needs |
|---|---|---|---|
| `gemini` | `gemini-3.6-flash` (Google) | cloud | `GEMINI_API_KEY` |
| `gpt` | `gpt-5.6` (OpenAI) | cloud | `OPENAI_API_KEY` |
| `qwen` | Qwen (open-weight) | GPU server | a local vLLM stack |

```bash
export GEMINI_API_KEY=...  # makes the Google/Gemini backbone available
export OPENAI_API_KEY=...  # makes the OpenAI/GPT backbone available

# Or point at an OpenAI-compatible vLLM stack you serve yourself:
export FH_QWEN_BASE_URL=http://gpu-box:8000/v1
export FH_QWEN_READER_BASE_URL=http://gpu-box:8001/v1   # the long-document reader
```

Pick a backbone per run with `--profile NAME`, or set the default with
`FH_PROFILE` or in `providers.json` (below).

### Configuration files

The defaults are built in; nothing needs to exist. To change them, put either
file in `~/.financeharness/` (or the directory named by `FH_CONFIG_DIR`). This
works the same for an installed copy and a source checkout. Each setting is
read in this order, later wins: built-in default, file, environment variable.

`providers.json` sets the default backbone and changes or adds profiles:

```json
{
  "default": "gpt",
  "profiles": {
    "gpt": { "model": "gpt-5.6" },
    "local": { "model": "my-model", "base_url": "http://127.0.0.1:8000/v1",
               "api_key_literal": "EMPTY", "reader_profile": "local" }
  }
}
```

`runtime.json` sets run limits and page reading. Only the keys to change are
needed:

```json
{
  "limits": { "max_rounds": 60 },
  "reader": { "max_chars": 200000 },
  "fetch": { "sec_user_agent": "Example Research admin@example.com" }
}
```

- `reader.max_chars`: characters of a page sent to the reader. `null` (the
  default) sizes it to the reader's context window (`context_window` in a
  profile, else `recovery.context_window`).
- `fetch.sec_user_agent` (or `FH_SEC_USER_AGENT`): User-Agent sent to sec.gov.
  The SEC asks automated clients for a name and a contact email.

A `runtime.json` that cannot be used is ignored as a whole. A profile that is
not valid is skipped; if it overrides a built-in profile, the built-in one is
kept. Each case gives one warning on stderr. The sections of `runtime.json` and
their defaults are the models in
[`financeharness/runtime/config.py`](financeharness/runtime/config.py).

### Run

One-shot research. The cited report goes to stdout, progress to stderr, so
stdout stays pipeable:

```bash
fh -p "Estimate NVDA's intrinsic value with a DCF."
fh -p "Apple's competitive position in 2026?" --mode research
fh -p "..." --profile gpt --save run.json  # switch backbone; persist trajectory
echo "What's AAPL's P/E?" | fh -p          # question piped via stdin
fh --list                                  # backbones + skills; confirms the install
```

Options: `--mode {auto|research|analytical}`, `--profile NAME`, `--reader NAME`,
`--save PATH`, `--quiet`. Exit code is `0` when the agent produced an answer,
`1` otherwise.

### Modes

Modes are prompt variants over a constant tool registry, so switching never
strands a trajectory. Set with `--mode`.

| Mode | Focus |
|---|---|
| `auto` | Full toolkit; the agent decides |
| `research` | Web-first deep research |
| `analytical` | Numbers-first: data and valuation tools, web for context |

A `-p` one-shot defaults to `research`.

## Tools

Seven tools are always visible; the rest sit in a catalog and are loaded on
demand with `load_tool`.

| Group | Tools |
|---|---|
| **Web research** | `search`, `visit`, `compose_citations` |
| **Core** | `calc`, `update_plan`, `load_tool`, `load_skill` |
| **Equity data** | `data_equity_reference`, `data_equity_prices`, `data_equity_fundamentals`, `data_equity_ratios`, `data_equity_comps`, `data_equity_estimates` |
| **Market data** | `data_market_rates`, `data_market_indices` |
| **Options data** | `data_options_chain` |
| **Valuation** | `compute_valuation_dcf`, `compute_valuation_dcf_sensitivity`, `compute_valuation_wacc` |
| **Risk** | `compute_risk_correlation`, `compute_risk_var`, `compute_risk_beta` |
| **Options** | `compute_options_black_scholes`, `compute_options_implied_vol` |

Company, market, and option-chain data are sourced through `yfinance`; option
quotes are delayed and carry no Greeks, which the options tools compute.
`data_equity_prices` returns `close` as traded and `adj_close` adjusted for
dividends. The risk tools take its `bars` whole, pair series by date and read
`adj_close`.

## Extend with skills

A **skill** is a `SKILL.md` (YAML frontmatter + a markdown body) that orchestrates
the existing tools into a reusable workflow, no code required. The model loads
one with `load_skill` when it fits the task. Bundled:

| Skill | What it does |
|---|---|
| `ticker-snapshot` | Quick structured overview of one equity (identity, ratios, price/trend). |
| `dcf-valuation` | Intrinsic value via DCF: fundamentals → CAPM/WACC → project and discount. |
| `relative-valuation` | Peer-median multiples applied to the company for an implied range. |
| `consensus-check` | Sell-side view (targets, estimates, ratings) corroborated against the web. |
| `equity-deep-dive` | The full workflow: qualitative picture + fundamentals + DCF + comps + consensus. |
| `options-pricing` | Option chain, risk-free rate and realized vol into Black-Scholes prices, Greeks and implied vol, against the market. |

Discovery runs in increasing precedence: bundled `financeharness/skills/` → a
project `./skills/` → `FH_SKILLS_DIR`. A project skill overrides a built-in by
name, and a malformed one is skipped rather than fatal.

## HTTP + SSE service (optional)

The package also ships an HTTP+SSE service for a remote client or programmatic
use; it is entirely separate from the CLI:

```bash
fh serve  # HTTP+SSE on 127.0.0.1:8080
```

## Benchmark

The **FinanceGym** benchmark and leaderboard are maintained at
[google-research/finance_harness](https://github.com/google-research/google-research/tree/master/finance_harness).

## Citation

If you find *FinanceHarness* helpful, please cite our work.

```bibtex
@misc{xiao2026financeharnessautonomousfinancialdeep,
      title={FinanceHarness: Autonomous Financial Deep Research Framework}, 
      author={Yijia Xiao and Rujun Han and Yanfei Chen and Zifeng Wang and Ke Jiang and Zhongying CuiZhu and Vishy Tirumalashetty and Wei Wang and Burak Gokturk and Tomas Pfister and Chen-Yu Lee},
      year={2026},
      eprint={2607.27853},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.27853}, 
}
```
