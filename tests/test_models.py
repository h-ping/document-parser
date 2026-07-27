import unittest

from document_parser.models import to_jsonable


class ModelTests(unittest.TestCase):
    def test_to_jsonable_preserves_explicit_null_values(self) -> None:
        value = {"target": None, "nested": {"optional": None}}

        self.assertEqual(to_jsonable(value), {"target": None, "nested": {"optional": None}})


if __name__ == "__main__":
    unittest.main()
