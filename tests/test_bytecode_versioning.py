"""§4.4 — Bytecode Versioning tests.

Exercises `src.backend.bytecode` version-management surfaces:
  - BytecodeVersion parsing (int, tuple, str, "v" prefix)
  - comparison operators (==, <, <=, >, >=)
  - CompatibilityStatus classification (EXACT, FORWARD_MINOR,
    BACKWARD, INCOMPATIBLE_MAJOR/FUTURE)
  - validate_bytecode_version: raises on incompatible major,
    accepts forward-minor & backward (forward-compatible handling)
  - clear error messages via format_version_error
  - is_bytecode_compatible (no exceptions)
  - legacy BYTECODE_VERSION int preserved for back-compat
"""
import unittest

from src.backend.bytecode import (
    BytecodeVersion,
    BYTECODE_VERSION,
    BYTECODE_VERSION_MAJOR,
    BYTECODE_VERSION_MINOR,
    SUPPORTED_BYTECODE_VERSION,
    CompatibilityStatus,
    parse_bytecode_version,
    check_bytecode_compatibility,
    is_bytecode_compatible,
    validate_bytecode_version,
    format_version_error,
    load_bytecode,
    UnsupportedBytecodeVersionError,
    Instruction,
    Opcode,
)


# ---------------------------------------------------------------------------
# BytecodeVersion parsing & basic methods
# ---------------------------------------------------------------------------

class TestBytecodeVersionParsing(unittest.TestCase):
    def test_from_int(self):
        v = BytecodeVersion.from_int(2)
        self.assertEqual(v.major, 2)
        self.assertEqual(v.minor, 0)

    def test_from_tuple_major_only(self):
        v = BytecodeVersion.from_tuple((3,))
        self.assertEqual(v.major, 3)
        self.assertEqual(v.minor, 0)

    def test_from_tuple_major_minor(self):
        v = BytecodeVersion.from_tuple((1, 5))
        self.assertEqual(v.major, 1)
        self.assertEqual(v.minor, 5)

    def test_from_str_plain(self):
        v = BytecodeVersion.from_str("1.5")
        self.assertEqual(v.major, 1)
        self.assertEqual(v.minor, 5)

    def test_from_str_v_prefix(self):
        v = BytecodeVersion.from_str("v2.3")
        self.assertEqual(v.major, 2)
        self.assertEqual(v.minor, 3)

    def test_from_str_major_only(self):
        v = BytecodeVersion.from_str("3")
        self.assertEqual(v.major, 3)
        self.assertEqual(v.minor, 0)

    def test_parse_dispatches_int(self):
        self.assertEqual(parse_bytecode_version(2).major, 2)

    def test_parse_dispatches_tuple(self):
        self.assertEqual(parse_bytecode_version((1, 4)).minor, 4)

    def test_parse_dispatches_str(self):
        v = parse_bytecode_version("1.0")
        self.assertEqual((v.major, v.minor), (1, 0))

    def test_parse_dispatches_bytecodeversion(self):
        v1 = BytecodeVersion(1, 2)
        v2 = parse_bytecode_version(v1)
        self.assertIsNot(v1, v2)  # parse() always returns a copy
        self.assertEqual(v1, v2)

    def test_parse_rejects_bool(self):
        with self.assertRaises(TypeError):
            BytecodeVersion.parse(True)

    def test_parse_rejects_negative(self):
        with self.assertRaises(ValueError):
            BytecodeVersion(-1, 0)

    def test_parse_rejects_unknown_type(self):
        with self.assertRaises(TypeError):
            BytecodeVersion.parse(3.14)

    def test_parse_rejects_empty_string(self):
        with self.assertRaises(ValueError):
            BytecodeVersion.from_str("")


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------

class TestBytecodeVersionComparison(unittest.TestCase):
    def test_equality_same_minor(self):
        self.assertEqual(BytecodeVersion(1, 0), BytecodeVersion(1, 0))

    def test_equality_with_int(self):
        # BytecodeVersion(1,0) compares equal to int 1
        self.assertEqual(BytecodeVersion(1, 0), 1)

    def test_equality_with_tuple(self):
        self.assertEqual(BytecodeVersion(2, 5), (2, 5))

    def test_equality_with_str(self):
        self.assertEqual(BytecodeVersion(1, 3), "1.3")

    def test_less_than_by_minor(self):
        self.assertTrue(BytecodeVersion(1, 0) < BytecodeVersion(1, 1))

    def test_less_than_by_major(self):
        self.assertTrue(BytecodeVersion(1, 9) < BytecodeVersion(2, 0))

    def test_greater_than(self):
        self.assertTrue(BytecodeVersion(2, 0) > BytecodeVersion(1, 9))

    def test_le_and_ge_same(self):
        self.assertTrue(BytecodeVersion(1, 0) <= BytecodeVersion(1, 0))
        self.assertTrue(BytecodeVersion(1, 0) >= BytecodeVersion(1, 0))

    def test_str_format(self):
        self.assertEqual(str(BytecodeVersion(1, 5)), "1.5")

    def test_repr_format(self):
        self.assertEqual(repr(BytecodeVersion(2, 3)),
                          "BytecodeVersion(2, 3)")

    def test_as_tuple(self):
        self.assertEqual(BytecodeVersion(3, 7).as_tuple(), (3, 7))

    def test_as_int_returns_major(self):
        self.assertEqual(BytecodeVersion(2, 9).as_int(), 2)

    def test_hashable(self):
        s = {BytecodeVersion(1, 0), BytecodeVersion(1, 0),
              BytecodeVersion(2, 0)}
        self.assertEqual(len(s), 2)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants(unittest.TestCase):
    def test_supported_version_is_major_one(self):
        self.assertEqual(SUPPORTED_BYTECODE_VERSION.major,
                          BYTECODE_VERSION_MAJOR)

    def test_supported_version_minor_zero(self):
        self.assertEqual(SUPPORTED_BYTECODE_VERSION.minor,
                          BYTECODE_VERSION_MINOR)

    def test_legacy_int_alias_matches_major(self):
        self.assertEqual(BYTECODE_VERSION, BYTECODE_VERSION_MAJOR)
        self.assertEqual(BYTECODE_VERSION, 1)

    def test_supported_bytecode_version_str(self):
        self.assertEqual(str(SUPPORTED_BYTECODE_VERSION), "1.0")


# ---------------------------------------------------------------------------
# Compatibility checking
# ---------------------------------------------------------------------------

class TestCompatibilityStatus(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(
            check_bytecode_compatibility(BytecodeVersion(1, 0)),
            CompatibilityStatus.EXACT
        )

    def test_exact_match_with_int(self):
        self.assertEqual(
            check_bytecode_compatibility(1),
            CompatibilityStatus.EXACT
        )

    def test_exact_match_with_str(self):
        self.assertEqual(
            check_bytecode_compatibility("1.0"),
            CompatibilityStatus.EXACT
        )

    def test_forward_minor(self):
        # Same major, higher minor → forward-compatible
        self.assertEqual(
            check_bytecode_compatibility(BytecodeVersion(1, 5)),
            CompatibilityStatus.FORWARD_MINOR
        )

    def test_backward_same_major_lower_minor(self):
        # Older minor (same major) → backward compatible
        # Note: with both at major=1, if requested minor < supported minor,
        # we go BACKWARD.  But supported is (1,0); requested (1,0) is exact,
        # so to test backward we need supported > 0.  Use exact same
        # interpretation: requested (1, 0) == supported (1, 0) => EXACT.
        # Use a request lower than supported — but supported is (1, 0).
        # A version like (0, 5) is a different major so it's BACKWARD.
        self.assertEqual(
            check_bytecode_compatibility(BytecodeVersion(0, 5)),
            CompatibilityStatus.BACKWARD
        )

    def test_incompatible_future_major(self):
        # Major higher than supported (1) → future incompatible
        self.assertEqual(
            check_bytecode_compatibility(BytecodeVersion(99, 0)),
            CompatibilityStatus.INCOMPATIBLE_FUTURE
        )

    def test_invalid_version_returns_incompatible_major(self):
        self.assertEqual(
            check_bytecode_compatibility("not-a-version"),
            CompatibilityStatus.INCOMPATIBLE_MAJOR
        )

    def test_invalid_type_returns_incompatible_major(self):
        self.assertEqual(
            check_bytecode_compatibility(None),
            CompatibilityStatus.INCOMPATIBLE_MAJOR
        )


# ---------------------------------------------------------------------------
# is_bytecode_compatible (non-raising boolean check)
# ---------------------------------------------------------------------------

class TestIsBytecodeCompatible(unittest.TestCase):
    def test_exact_is_compatible(self):
        self.assertTrue(is_bytecode_compatible("1.0"))

    def test_forward_minor_is_compatible(self):
        self.assertTrue(is_bytecode_compatible("1.7"))

    def test_backward_is_compatible(self):
        self.assertTrue(is_bytecode_compatible("0.5"))

    def test_future_major_is_incompatible(self):
        self.assertFalse(is_bytecode_compatible("2.0"))

    def test_invalid_input_is_incompatible(self):
        self.assertFalse(is_bytecode_compatible("garbage"))


# ---------------------------------------------------------------------------
# validate_bytecode_version
# ---------------------------------------------------------------------------

class TestValidateBytecodeVersion(unittest.TestCase):
    def test_valid_dict_returns_true(self):
        self.assertTrue(validate_bytecode_version(
            {"bytecode_version": 1}))

    def test_valid_dict_with_str_version_returns_true(self):
        self.assertTrue(validate_bytecode_version(
            {"bytecode_version": "1.0"}))

    def test_forward_minor_returns_true(self):
        self.assertTrue(validate_bytecode_version(
            {"bytecode_version": "1.5"}))

    def test_backward_returns_true(self):
        self.assertTrue(validate_bytecode_version(
            {"bytecode_version": "0.9"}))

    def test_future_major_raises(self):
        with self.assertRaises(UnsupportedBytecodeVersionError):
            validate_bytecode_version(
                {"bytecode_version": "2.0"})

    def test_legacy_int_format_supported(self):
        # Old serialized bytecode uses int version=1
        self.assertTrue(validate_bytecode_version(
            {"bytecode_version": 1, "instructions": []}))

    def test_legacy_int_future_raises(self):
        with self.assertRaises(UnsupportedBytecodeVersionError):
            validate_bytecode_version(
                {"bytecode_version": 99})

    def test_missing_version_field_defaults_to_zero(self):
        # Default 0 → backward-compatible (treated as 0.0)
        self.assertTrue(validate_bytecode_version({"instructions": []}))

    def test_non_dict_returns_true(self):
        # If passed non-dict, just return True (nothing to validate)
        self.assertTrue(validate_bytecode_version(None))
        self.assertTrue(validate_bytecode_version("hello"))


# ---------------------------------------------------------------------------
# Error message formatting
# ---------------------------------------------------------------------------

class TestFormatVersionError(unittest.TestCase):
    def test_message_includes_requested_version(self):
        msg = format_version_error("2.5")
        self.assertIn("2.5", msg)

    def test_message_includes_supported_version(self):
        msg = format_version_error("2.5")
        self.assertIn("1.0", msg)

    def test_message_actionable_advice(self):
        msg = format_version_error("2.0")
        # Should hint at upgrading or recompiling
        self.assertTrue("Upgrade" in msg or "recompile" in msg)

    def test_message_works_with_bytecodeversion(self):
        msg = format_version_error(BytecodeVersion(3, 1))
        self.assertIn("3.1", msg)


# ---------------------------------------------------------------------------
# Compatibility with existing legacy API
# ---------------------------------------------------------------------------

class TestLegacyAPIBackwardsCompat(unittest.TestCase):
    def test_bytecode_version_int_constant(self):
        # Old code references BYTECODE_VERSION as an int
        self.assertEqual(BYTECODE_VERSION, 1)
        self.assertIsInstance(BYTECODE_VERSION, int)

    def test_validate_raises_on_int_99(self):
        # Test from test_nova_fixes.py F12 case
        invalid_data = {"bytecode_version": 99, "instructions": []}
        with self.assertRaises(UnsupportedBytecodeVersionError):
            validate_bytecode_version(invalid_data)

    def test_validate_accepts_int_1(self):
        # Test from test_nova_fixes.py F12 case
        valid_data = {"bytecode_version": 1, "instructions": []}
        self.assertTrue(validate_bytecode_version(valid_data))


# ---------------------------------------------------------------------------
# load_bytecode — forward-compatible loading with status
# ---------------------------------------------------------------------------

class TestLoadBytecode(unittest.TestCase):
    def test_load_exact_version_returns_instructions(self):
        data = {
            "bytecode_version": 1,
            "instructions": [
                {"opcode": "LOAD_CONST", "arg": 42, "line": 1},
                {"opcode": "HALT", "arg": None, "line": 2},
            ],
        }
        instrs, status = load_bytecode(data)
        self.assertEqual(len(instrs), 2)
        self.assertEqual(status, CompatibilityStatus.EXACT)
        self.assertEqual(instrs[0].opcode, "LOAD_CONST")
        self.assertEqual(instrs[0].arg, 42)

    def test_load_forward_minor_returns_instructions(self):
        data = {
            "bytecode_version": "1.5",
            "instructions": [
                {"opcode": "HALT", "arg": None, "line": 1},
            ],
        }
        instrs, status = load_bytecode(data)
        self.assertEqual(len(instrs), 1)
        self.assertEqual(status, CompatibilityStatus.FORWARD_MINOR)

    def test_load_backward_returns_instructions(self):
        data = {
            "bytecode_version": "0.9",
            "instructions": [],
        }
        instrs, status = load_bytecode(data)
        self.assertEqual(status, CompatibilityStatus.BACKWARD)

    def test_load_future_major_raises(self):
        data = {
            "bytecode_version": "2.0",
            "instructions": [],
        }
        with self.assertRaises(UnsupportedBytecodeVersionError):
            load_bytecode(data)

    def test_load_non_dict_returns_empty(self):
        instrs, status = load_bytecode(None)
        self.assertEqual(instrs, [])
        self.assertEqual(status, "exact")

    def test_load_empty_instructions(self):
        data = {"bytecode_version": 1, "instructions": []}
        instrs, status = load_bytecode(data)
        self.assertEqual(instrs, [])
        self.assertEqual(status, CompatibilityStatus.EXACT)

    def test_load_missing_instructions_key(self):
        data = {"bytecode_version": 1}
        instrs, status = load_bytecode(data)
        self.assertEqual(instrs, [])

    def test_load_preserves_line_info(self):
        data = {
            "bytecode_version": 1,
            "instructions": [
                {"opcode": "LOAD_CONST", "arg": 1, "line": 10},
            ],
        }
        instrs, _ = load_bytecode(data)
        self.assertEqual(instrs[0].line, 10)


if __name__ == "__main__":
    unittest.main()
