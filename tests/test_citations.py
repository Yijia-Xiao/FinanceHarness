"""Citation handling: the source list, inline marker checks and the references section."""

import pytest

from financeharness.runtime.citations import validate_and_append_references
from financeharness.runtime.dispatch import dispatch
from financeharness.runtime.tool_registry import ToolRegistry
from financeharness.tools.research import FetchCache, build_citations_spec, citation_finalizer
from tests.support import run


@pytest.fixture
def cache():
  cache = FetchCache()
  cache.add_citation("https://example.com/10k", "Annual report")
  cache.add_citation("https://example.com/call", "Earnings call")
  return cache


REFERENCES = (
    "## References\n\n"
    "[1] Annual report\n    https://example.com/10k\n"
    "[2] Earnings call\n    https://example.com/call\n"
)


def test_sources_are_numbered_in_visit_order_without_duplicates():
  cache = FetchCache()
  cache.set_title("https://example.com/b", "Found by search")
  assert cache.add_citation("https://example.com/a", "First") == 1
  assert cache.add_citation("https://example.com/b") == 2
  assert cache.add_citation("https://example.com/a", "Seen again") == 1
  assert cache.add_citation("https://example.com/c") == 3
  assert [(c.index, c.title) for c in cache.citations] == [
      (1, "First"),
      (2, "Found by search"),
      (3, "https://example.com/c"),
  ]


def test_references_are_appended_and_unknown_markers_removed(cache):
  report = "Revenue rose 12% [1]. Margins fell [2, 5]. Guidance was cut [7]."

  text, stats = validate_and_append_references(report, cache.citations)

  assert text == "Revenue rose 12% [1]. Margins fell [2]. Guidance was cut.\n\n" + REFERENCES
  assert stats == {
      "n_refs": 2,
      "n_inline_citations": 2,
      "cited_indices": [1, 2],
      "n_orphans_stripped": 2,
      "unused_indices": [],
  }


def test_sources_never_cited_are_reported_as_unused(cache):
  text, stats = validate_and_append_references("Margins fell [2].", cache.citations)
  assert stats["cited_indices"] == [2]
  assert stats["unused_indices"] == [1]
  assert text.endswith(REFERENCES)


@pytest.mark.parametrize(
    "heading", ["## References", "### references", "**References**", "References:"]
)
def test_reference_list_written_by_the_model_is_replaced(cache, heading):
  report = f"Revenue rose [1].\n\n{heading}\n[1] Made up source\n[2] Another one"

  text, _ = validate_and_append_references(report, cache.citations)

  assert text == "Revenue rose [1].\n\n" + REFERENCES


def test_report_without_sources_gets_no_references_and_no_markers():
  text, stats = validate_and_append_references("Rates are unchanged [1].", [])
  assert text == "Rates are unchanged.\n"
  assert stats["n_refs"] == 0
  assert stats["n_orphans_stripped"] == 1


def test_leaked_end_of_report_note_is_removed(cache):
  report = (
      "Revenue rose [1].\n\n"
      "[End of your report. The harness appends the References section automatically.]"
  )
  text, _ = validate_and_append_references(report, cache.citations)
  assert text == "Revenue rose [1].\n\n" + REFERENCES


def test_sections_after_a_line_reading_references_are_kept(cache):
  report = (
      "Revenue rose [1].\n\n"
      "References\nto the prior filing recur through the call [2].\n\n"
      "## Outlook\n\nManagement expects growth to continue."
  )
  text, _ = validate_and_append_references(report, cache.citations)
  assert "## Outlook\n\nManagement expects growth to continue." in text


def test_text_after_a_leaked_end_of_report_note_is_kept(cache):
  report = (
      "Revenue rose [1].\n"
      "[End of output. The system appends references.]\n\n"
      "## Outlook\n\nGrowth continues."
  )
  text, _ = validate_and_append_references(report, cache.citations)
  assert text.startswith("Revenue rose [1].\n\n## Outlook\n\nGrowth continues.\n")


def test_sentence_that_resembles_the_end_note_is_kept(cache):
  report = "At the end of report. The appendix appends the references list here [1]."
  text, _ = validate_and_append_references(report, cache.citations)
  assert text.startswith(report + "\n")


@pytest.mark.parametrize(
    "listing",
    [
        "[1] Annual report\n[2] Earnings call",
        "1. Annual report\n2. Earnings call",
        "- Annual report\n- Earnings call",
        "https://example.com/10k",
    ],
)
def test_bare_references_line_followed_by_a_list_is_replaced(cache, listing):
  text, _ = validate_and_append_references(
      "Revenue rose [1].\n\nReferences\n" + listing, cache.citations
  )
  assert text == "Revenue rose [1].\n\n" + REFERENCES


def test_markers_attached_to_a_word_are_checked(cache):
  text, stats = validate_and_append_references(
      "Margin[2,9] widened and revenue rose[1][7].", cache.citations
  )
  assert text.startswith("Margin[2] widened and revenue rose[1].\n")
  assert stats["cited_indices"] == [1, 2]
  assert stats["n_orphans_stripped"] == 2


def test_bracketed_numbers_inside_code_are_left_alone(cache):
  report = "Take `bars[0]` first [1], drop [5].\n\n```\nrows[3] = rows[0]\n```"
  text, stats = validate_and_append_references(report, cache.citations)
  assert text.startswith("Take `bars[0]` first [1], drop.\n\n```\nrows[3] = rows[0]\n```\n")
  assert stats["cited_indices"] == [1]
  assert stats["n_orphans_stripped"] == 1


def test_finalizer_reads_the_sources_known_at_the_end_of_the_run():
  cache = FetchCache()
  finalize = citation_finalizer(cache)
  cache.add_citation("https://example.com/10k", "Annual report")

  text, stats = finalize("Revenue rose [1].")

  assert text.endswith("[1] Annual report\n    https://example.com/10k\n")
  assert stats["n_refs"] == 1


def test_compose_citations_lists_sources_for_the_model(cache):
  reg = ToolRegistry()
  reg.register(build_citations_spec(cache))

  result = run(dispatch("compose_citations", {}, registry=reg))

  assert result.markdown.startswith(
      "[1] Annual report\n    https://example.com/10k\n[2] Earnings call"
  )
  assert result.structured == {
      "count": 2,
      "citations": ["https://example.com/10k", "https://example.com/call"],
  }


def test_compose_citations_says_so_when_nothing_was_read():
  reg = ToolRegistry()
  reg.register(build_citations_spec(FetchCache()))
  result = run(dispatch("compose_citations", {}, registry=reg))
  assert result.ok
  assert result.structured == {"count": 0}
  assert "No sources have been read yet" in result.markdown
