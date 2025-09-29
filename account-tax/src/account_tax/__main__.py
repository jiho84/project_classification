"""Entry point for running the pipeline."""

import sys
from pathlib import Path
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession


def main():
    """Run the pipeline."""
    # Configure project
    project_path = Path(__file__).parent.parent.parent
    configure_project("account_tax")
    
    # Run pipeline
    with KedroSession.create(project_path=project_path) as session:
        session.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
