import unittest
from typing import TypedDict

from document_parser.global_reconciliation import GlobalReconciliationError, validate_and_finalize_reconciliation


class GlobalReconciliationTests(unittest.TestCase):
    def test_reconciliation_preserves_tables_and_removes_duplicate_field_proposals(self) -> None:
        proposals = {
            "fields": [
                _field("product.name", "atom_1", "测试产品"),
                _field("product.name", "atom_2", "测试产品"),
            ],
            "entities": [],
            "tables": [{"table_id": "nutrition_1"}],
            "requirements": [],
            "ignored_nodes": [],
            "unknown_nodes": ["atom_3"],
            "layout_candidate_decisions": [],
            "node_scope_decisions": [],
        }
        reconciled = {
            **proposals,
            "fields": [proposals["fields"][0]],
            "tables": [],
            "unknown_nodes": ["atom_4"],
        }

        finalized, report = validate_and_finalize_reconciliation(proposals, reconciled)

        self.assertEqual(len(finalized["fields"]), 1)
        self.assertEqual(finalized["tables"], proposals["tables"])
        self.assertEqual(finalized["unknown_nodes"], ["atom_3", "atom_4"])
        self.assertEqual(report["removed_field_count"], 1)

    def test_reconciliation_rejects_ranges_not_present_in_proposals(self) -> None:
        proposals = {
            "fields": [_field("product.name", "atom_1", "测试产品")],
            "entities": [],
            "tables": [],
            "requirements": [],
            "ignored_nodes": [],
            "unknown_nodes": [],
            "layout_candidate_decisions": [],
            "node_scope_decisions": [],
        }
        reconciled = {**proposals, "fields": [_field("product.name", "atom_1", "伪造文本")]}

        with self.assertRaises(GlobalReconciliationError):
            validate_and_finalize_reconciliation(proposals, reconciled)

    def test_reconciliation_rejects_range_borrowed_from_another_semantic_key(self) -> None:
        proposals = {
            "fields": [
                _field("product.name", "atom_1", "测试产品"),
                _field("product.net_content", "atom_2", "100克"),
            ],
            "entities": [],
        }
        reconciled = {
            "fields": [_field("product.name", "atom_2", "100克")],
            "entities": [],
        }

        with self.assertRaises(GlobalReconciliationError):
            validate_and_finalize_reconciliation(proposals, reconciled)

    def test_reconciliation_validates_fields_nested_under_entities(self) -> None:
        proposals = {
            "fields": [_field("manufacturer.name", "atom_1", "甲公司")],
            "entities": [],
        }
        reconciled = {
            "fields": [],
            "entities": [
                {
                    "entity_id": "manufacturer_1",
                    "fields": [_field("manufacturer.address", "atom_1", "甲公司")],
                }
            ],
        }

        with self.assertRaises(GlobalReconciliationError):
            validate_and_finalize_reconciliation(proposals, reconciled)


class FieldProposal(TypedDict):
    semantic_key: str
    display_name: str
    field_type: str
    span_id: str
    start_offset: int
    end_offset: int
    text: str
    confidence: float
    entity_id: None
    section_id: None


def _field(semantic_key: str, span_id: str, text: str) -> FieldProposal:
    return {
        "semantic_key": semantic_key,
        "display_name": semantic_key,
        "field_type": "string",
        "span_id": span_id,
        "start_offset": 0,
        "end_offset": len(text),
        "text": text,
        "confidence": 0.9,
        "entity_id": None,
        "section_id": None,
    }
