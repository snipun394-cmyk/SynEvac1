import tempfile
import unittest

from calibration_studio.git_provenance import capture_git_provenance


class CaptureGitProvenanceInRealRepoTests(unittest.TestCase):

    def test_commit_hash_is_a_well_formed_sha(self):

        provenance = capture_git_provenance()

        self.assertIsNotNone(provenance.commit_hash)
        self.assertEqual(len(provenance.commit_hash), 40)
        int(provenance.commit_hash, 16)  # raises ValueError if not hex

    def test_dirty_is_a_real_boolean_not_none_in_a_real_repo(self):

        provenance = capture_git_provenance()

        self.assertIsInstance(provenance.dirty, bool)

    def test_to_dict_round_trips(self):

        provenance = capture_git_provenance()

        as_dict = provenance.to_dict()
        self.assertEqual(as_dict["commit_hash"], provenance.commit_hash)
        self.assertEqual(as_dict["dirty"], provenance.dirty)


class CaptureGitProvenanceOutsideAnyRepoTests(unittest.TestCase):

    def test_non_git_directory_degrades_honestly_to_none_none(self):

        with tempfile.TemporaryDirectory() as tmpdir:

            provenance = capture_git_provenance(cwd=tmpdir)

            self.assertIsNone(provenance.commit_hash)
            self.assertIsNone(provenance.dirty)


if __name__ == "__main__":
    unittest.main()
