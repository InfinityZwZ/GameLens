"""Complete report entry point; readable rendering lives in report_prose."""
from gamelens.analysis.report_prose import write_complete_report, refresh_existing_reports


if __name__=="__main__":
    from gamelens.core.paths import PROJECT_ROOT
    print("已重排报告："+"、".join(refresh_existing_reports(PROJECT_ROOT)))
