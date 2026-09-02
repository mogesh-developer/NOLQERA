class NOLQERAPipelineError(Exception):
    """
    Base exception for all NOLQERA pipeline errors.
    """


class PipelineConfigurationError(NOLQERAPipelineError):
    """
    Raised when pipeline configuration is invalid.
    """


class PipelineExecutionError(NOLQERAPipelineError):
    """
    Raised when a pipeline stage fails during execution.
    """


class PipelineStageError(PipelineExecutionError):
    """
    Raised when a specific pipeline stage fails.
    """

    def __init__(
        self,
        stage: str,
        message: str,
    ) -> None:
        self.stage = stage
        self.message = message

        super().__init__(
            f"Pipeline stage '{stage}' failed: {message}"
        )