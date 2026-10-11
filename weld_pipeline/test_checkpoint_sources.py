import tempfile
import unittest
from pathlib import Path
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources


class SourceEvidenceTest(unittest.TestCase):
    def test_historical_entrypoint_requires_matching_archive(self):
        with tempfile.TemporaryDirectory(dir='.codex-local/tmp') as d:
            root=Path(d);source=root/'predict_typed.py';source.write_text('old')
            state={'source_hashes':{str(source):sha256(source)}}
            source.write_text('new');archive=root/'source_snapshot';archive.mkdir()
            with self.assertRaises(ValueError):verify_cached_inference_sources(state,root/'best.pt')
            (archive/source.name).write_text('old')
            self.assertEqual(len(verify_cached_inference_sources(state,root/'best.pt')),1)

    def test_runtime_dependency_cannot_use_archive(self):
        with tempfile.TemporaryDirectory(dir='.codex-local/tmp') as d:
            root=Path(d);source=root/'invariant_surface_model.py';source.write_text('old')
            state={'source_hashes':{str(source):sha256(source)}}
            archive=root/'source_snapshot';archive.mkdir();(archive/source.name).write_text('old')
            source.write_text('new')
            with self.assertRaises(ValueError):verify_cached_inference_sources(state,root/'best.pt')


if __name__=='__main__':unittest.main()
