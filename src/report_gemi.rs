use crate::linkage_gemi::ApiLinkage;

pub fn generate_linkage_report(linkages: &[ApiLinkage]) -> String {
    let mut report = String::new();

    report.push_str("API Linkage Report\n");
    report.push_str("===================\n\n");

    if linkages.is_empty() {
        report.push_str("No API linkages found.\n");
        return report;
    }

    report.push_str(&format!("Found {} potential API linkages.\n\n", linkages.len()));

    for (i, linkage) in linkages.iter().enumerate() {
        report.push_str(&format!("--- Linkage {}\n", i + 1));
        report.push_str(&format!("  Type: {:?}\n", linkage.linkage_type));
        report.push_str(&format!("  Score: {:.4}\n", linkage.similarity_score));
        report.push_str("  Frontend Call:\n");
        report.push_str(&format!("    File: {}\n", linkage.frontend_call.file_path));
        report.push_str(&format!("    Lines: {}- {}\n", linkage.frontend_call.start_line, linkage.frontend_call.end_line));
        report.push_str(&format!("    Path: {}\n", linkage.frontend_call.path));
        if let Some(method) = &linkage.frontend_call.method {
            report.push_str(&format!("    Method: {}\n", method));
        }
        report.push_str(&format!("    Context:\n{}\n", linkage.frontend_call.context_snippet));
        report.push_str("  Backend Endpoint:\n");
        report.push_str(&format!("    File: {}\n", linkage.backend_endpoint.file_path));
        report.push_str(&format!("    Lines: {}- {}\n", linkage.backend_endpoint.start_line, linkage.backend_endpoint.end_line));
        report.push_str(&format!("    Method: {}\n", linkage.backend_endpoint.method));
        report.push_str(&format!("    Path: {}\n", linkage.backend_endpoint.path));
        report.push_str(&format!("    Handler: {}\n", linkage.backend_endpoint.handler));
        report.push_str("\n");
    }

    report
}
