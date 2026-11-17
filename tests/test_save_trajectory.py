"""save_trajectory writes UTF-8 whatever the platform's default text encoding is."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from financeharness.research import save_trajectory


def _cp1252_default(encoding, stacklevel=2):
  # What an encoding-less pathlib write resolves to on a Windows cp1252 locale.
  return encoding or "cp1252"


class SaveTrajectoryEncodingTest(unittest.TestCase):

  def test_round_trips_text_outside_cp1252(self):
    traj = {
        "question": "AAPL P/E?",
        "prediction": "P/E 31.2 → fair value ≥ $210 (苹果) 🚀",
        "citations": [{"index": 1, "url": "https://example.com", "title": "Δ ≈ 0.5"}],
    }
    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch.object(io, "text_encoding", _cp1252_default),
    ):
      path = save_trajectory(traj, Path(tmp) / "runs" / "traj.json")
      self.assertEqual(json.loads(path.read_text(encoding="utf-8")), traj)


if __name__ == "__main__":
  unittest.main()
