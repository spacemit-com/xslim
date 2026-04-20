"""Regression tests for ONNX opset-24 operator domain handling."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xslim.defs import MIN_ONNX_OPSET_VERSION, resolve_operator_domain
from xslim.ppq_decorator.ppq.parser.onnx_exporter import (AttentionExporter,
                                                          OOSExporter)


class TestOnnxOperatorSupport(unittest.TestCase):
    """Validate standard ONNX operator support checks used during export."""

    def test_resolve_operator_domain_prefers_ai_onnx_when_supported(self):
        self.assertEqual(MIN_ONNX_OPSET_VERSION, 24)
        self.assertIsNone(resolve_operator_domain("Gelu", MIN_ONNX_OPSET_VERSION))
        self.assertIsNone(resolve_operator_domain("QLinearConv", MIN_ONNX_OPSET_VERSION))
        self.assertIsNone(resolve_operator_domain("Attention", MIN_ONNX_OPSET_VERSION))

    def test_resolve_operator_domain_keeps_custom_domain_when_needed(self):
        self.assertEqual(
            resolve_operator_domain("DynamicQuantizeMatMul", MIN_ONNX_OPSET_VERSION),
            "com.microsoft",
        )
        self.assertEqual(
            resolve_operator_domain("QAttention", MIN_ONNX_OPSET_VERSION),
            "com.microsoft",
        )

    def test_oos_exporter_drops_custom_domain_for_standardized_ops(self):
        op = mock.Mock(type="QLinearConv", attributes={"domain": "com.microsoft"})

        exported_op = OOSExporter().export(op, graph=None)

        self.assertIs(exported_op, op)
        self.assertNotIn("domain", exported_op.attributes)

    def test_exporters_preserve_custom_domain_for_unsupported_ops(self):
        qattention = mock.Mock(type="QAttention", attributes={})
        exported_qattention = OOSExporter().export(qattention, graph=None)
        self.assertEqual(exported_qattention.attributes["domain"], "com.microsoft")

        attention = mock.Mock(type="Attention", attributes={"domain": "com.microsoft"})
        exported_attention = AttentionExporter().export(attention, graph=None)
        self.assertNotIn("domain", exported_attention.attributes)


if __name__ == "__main__":
    unittest.main()
