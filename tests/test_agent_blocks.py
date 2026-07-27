import unittest

from document_parser.agent_blocks import (
    build_agent_blocks,
    build_block_context,
    important_unconsumed_span_ids,
    merge_agent_plan_bodies,
    merge_block_retry_body,
    merge_schema_bodies,
    schema_induction_spans,
)
from document_parser.models import TextSpan


class AgentBlockTests(unittest.TestCase):
    def test_schema_induction_spans_keeps_late_important_anchor(self) -> None:
        spans = [TextSpan(f"span_{index:04d}", 1, f"普通文字{index}", "pdf_char_atom") for index in range(260)]
        spans[-1] = TextSpan("span_0259", 1, "生产许可证编号：SC123", "pdf_char_atom")

        selected = schema_induction_spans(spans, max_spans=240)

        self.assertEqual(len(selected), 240)
        self.assertIn("span_0259", {span.span_id for span in selected})

    def test_every_span_has_one_primary_block_and_large_pages_are_chunked(self) -> None:
        spans = [TextSpan(f"span_{index:04d}", 1, f"普通标签文字{index}", "pdf_char_atom") for index in range(170)]

        artifact = build_agent_blocks(spans, {"table_candidates": [], "side_marker_candidates": []})

        primary_ids = [span_id for block in artifact["blocks"] for span_id in block["source_span_ids"]]
        self.assertEqual(set(primary_ids), {span.span_id for span in spans})
        self.assertEqual(len(primary_ids), len(set(primary_ids)))
        self.assertEqual(artifact["source_span_coverage_rate"], 1.0)
        self.assertGreater(len(artifact["blocks"]), 1)
        self.assertLessEqual(max(block["span_count"] for block in artifact["blocks"]), 160)

    def test_builds_content_producer_nutrition_and_other_blocks(self) -> None:
        spans = [
            TextSpan("content", 1, "内容物：红豆粽", "pdf_char_atom"),
            TextSpan("ingredient", 1, "配料表：糯米、红豆", "pdf_char_atom"),
            TextSpan("producer", 1, "受托方：甲公司", "pdf_char_atom"),
            TextSpan("nutrition", 1, "营养成分表", "pdf_char_atom"),
            TextSpan("marker", 1, "标", "pdf_char_atom"),
        ]
        layout = {
            "table_candidates": [
                {"table_candidate_id": "producer_1", "table_type": "producer_info_repeated_rows", "source_span_ids": ["producer"]},
                {"table_candidate_id": "nutrition_1", "table_type": "nutrition_facts", "source_span_ids": ["nutrition"]},
            ],
            "side_marker_candidates": [{"source_span_ids": ["marker"]}],
        }

        artifact = build_agent_blocks(spans, layout)

        block_types = {block["block_type"] for block in artifact["blocks"]}
        self.assertEqual(block_types, {"content_item", "producer_group", "nutrition_table", "other_printed_label"})

    def test_context_is_local_and_body_merges_preserve_repeated_fields(self) -> None:
        blocks = build_agent_blocks(
            [TextSpan("a", 1, "受托方：甲", "pdf_char_atom"), TextSpan("b", 2, "受托方：乙", "pdf_char_atom")],
            {"table_candidates": [], "side_marker_candidates": []},
        )["blocks"]
        context = build_block_context(blocks[0], {"regions": [{"page": 1}], "table_candidates": [], "label_text_scope": {"rule": "printed"}})
        merged_schema = merge_schema_bodies(
            [
                {"field_definitions": [{"semantic_key": "manufacturer.name", "source_span_ids": ["a"]}]},
                {"field_definitions": [{"semantic_key": "manufacturer.name", "source_span_ids": ["b"]}]},
            ]
        )
        merged_plan = merge_agent_plan_bodies(
            [
                {"fields": [{"semantic_key": "manufacturer.name", "span_id": "a"}], "entities": [], "tables": [], "requirements": [], "ignored_nodes": [], "unknown_nodes": [], "layout_candidate_decisions": []},
                {"fields": [{"semantic_key": "manufacturer.name", "span_id": "b"}], "entities": [], "tables": [], "requirements": [], "ignored_nodes": [], "unknown_nodes": [], "layout_candidate_decisions": []},
            ]
        )

        self.assertEqual(context["agent_block"]["block_id"], blocks[0]["block_id"])
        self.assertEqual(context["regions"], [{"page": 1}])
        self.assertEqual(len(merged_schema["field_definitions"][0]["source_span_ids"]), 2)
        self.assertEqual(len(merged_plan["fields"]), 2)

    def test_retry_focuses_unconsumed_important_anchors_and_deduplicates_ranges(self) -> None:
        spans = [
            TextSpan("name", 1, "内容物：红豆粽", "pdf_char_atom"),
            TextSpan("ingredients", 1, "配料表：糯米、红豆", "pdf_char_atom"),
            TextSpan("note", 1, "普通说明", "pdf_char_atom"),
        ]
        primary = {"fields": [{"semantic_key": "content_item.name", "span_id": "name", "start_offset": 0, "end_offset": 7}]}
        retry = {
            "fields": [
                {"semantic_key": "content_item.name", "span_id": "name", "start_offset": 0, "end_offset": 7},
                {"semantic_key": "content_item.ingredients", "span_id": "ingredients", "start_offset": 0, "end_offset": 10},
            ]
        }

        self.assertEqual(important_unconsumed_span_ids(primary, spans), ["ingredients"])
        self.assertEqual(len(merge_block_retry_body(primary, retry)["fields"]), 2)

    def test_retry_preserves_distinct_ranges_for_same_semantic_key_and_entity(self) -> None:
        primary = {
            "fields": [
                {
                    "semantic_key": "custom.other_label_text",
                    "entity_id": "product_001",
                    "span_id": "line",
                    "start_offset": 0,
                    "end_offset": 3,
                }
            ]
        }
        retry = {
            "fields": [
                {
                    "semantic_key": "custom.other_label_text",
                    "entity_id": "product_001",
                    "span_id": "line",
                    "start_offset": 3,
                    "end_offset": 6,
                }
            ]
        }

        merged = merge_block_retry_body(primary, retry)

        self.assertEqual([(item["start_offset"], item["end_offset"]) for item in merged["fields"]], [(0, 3), (3, 6)])


if __name__ == "__main__":
    unittest.main()
