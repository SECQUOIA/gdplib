import logging

import pyomo.environ as pyo

DEPRECATION_PATTERNS = (
    "implicitly casting",
    "Implicit conversion of the Boolean indicator_var",
    "Using __getitem__ to return a set value from its (ordered) position",
    "associate_binary_var",
)


def _pyomo_warning_messages(caplog):
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name.startswith("pyomo")
    )


def _assert_no_known_pyomo_gdp_deprecations(caplog):
    messages = _pyomo_warning_messages(caplog)
    for pattern in DEPRECATION_PATTERNS:
        assert pattern not in messages


def test_gdp_col_build_uses_boolean_indicator_values(caplog):
    import gdplib.gdp_col

    caplog.set_level(logging.WARNING)

    gdplib.gdp_col.build_model()

    _assert_no_known_pyomo_gdp_deprecations(caplog)


def test_med_term_purchasing_uses_ordered_set_at(caplog):
    import gdplib.med_term_purchasing

    caplog.set_level(logging.WARNING)

    gdplib.med_term_purchasing.build_model()

    _assert_no_known_pyomo_gdp_deprecations(caplog)


def test_small_batch_bigm_avoids_boolean_indicator_conversion(caplog):
    import gdplib.small_batch

    caplog.set_level(logging.WARNING)
    model = gdplib.small_batch.build_model()

    pyo.TransformationFactory("gdp.bigm").apply_to(model)

    _assert_no_known_pyomo_gdp_deprecations(caplog)


def test_cstr_bigm_avoids_boolean_association_deprecations(caplog):
    import gdplib.cstr

    caplog.set_level(logging.WARNING)
    model = gdplib.cstr.build_model()

    pyo.TransformationFactory("gdp.bigm").apply_to(model)

    _assert_no_known_pyomo_gdp_deprecations(caplog)
