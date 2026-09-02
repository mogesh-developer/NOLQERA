def test_pipeline_package_exports_pipeline():
    from nolqera.intelligence.pipeline import (
        NOLQERAPipeline,
    )

    assert NOLQERAPipeline is not None


def test_pipeline_package_exports_configuration():
    from nolqera.intelligence.pipeline import (
        PipelineConfig,
    )

    assert PipelineConfig is not None


def test_pipeline_package_exports_models():
    from nolqera.intelligence.pipeline import (
        PipelineMetadata,
        PipelineResult,
    )

    assert PipelineMetadata is not None
    assert PipelineResult is not None


def test_pipeline_package_exports_exceptions():
    from nolqera.intelligence.pipeline import (
        NOLQERAPipelineError,
        PipelineConfigurationError,
        PipelineExecutionError,
        PipelineStageError,
    )

    assert NOLQERAPipelineError is not None
    assert PipelineConfigurationError is not None
    assert PipelineExecutionError is not None
    assert PipelineStageError is not None


def test_pipeline_package_exports_integration_helpers():
    from nolqera.intelligence.pipeline import (
        run_pipeline,
        create_default_configured_pipeline,
    )

    assert run_pipeline is not None
    assert create_default_configured_pipeline is not None


def test_root_package_exports_pipeline():
    from nolqera import (
        NOLQERAPipeline,
        PipelineConfig,
        PipelineMetadata,
        PipelineResult,
    )

    assert NOLQERAPipeline is not None
    assert PipelineConfig is not None
    assert PipelineMetadata is not None
    assert PipelineResult is not None


def test_root_package_exports_pipeline_exceptions():
    from nolqera import (
        NOLQERAPipelineError,
        PipelineConfigurationError,
        PipelineExecutionError,
        PipelineStageError,
    )

    assert NOLQERAPipelineError is not None
    assert PipelineConfigurationError is not None
    assert PipelineExecutionError is not None
    assert PipelineStageError is not None


def test_root_package_exports_pipeline_helpers():
    from nolqera import (
        run_pipeline,
        create_default_configured_pipeline,
    )

    assert run_pipeline is not None
    assert create_default_configured_pipeline is not None


def test_public_pipeline_class_is_same_object():
    from nolqera import NOLQERAPipeline
    from nolqera.intelligence.pipeline import (
        NOLQERAPipeline as PipelineFromModule,
    )

    assert NOLQERAPipeline is PipelineFromModule


def test_public_config_is_same_object():
    from nolqera import PipelineConfig
    from nolqera.intelligence.pipeline import (
        PipelineConfig as ConfigFromModule,
    )

    assert PipelineConfig is ConfigFromModule